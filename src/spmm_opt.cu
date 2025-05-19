#include "spmm_opt.h"

const int TILE_SIZE = 32;
const int DENSE_BLOCK_SIZE = 32;

__global__ void spmm_kernel_dense_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE, int *dense_bid2posi, int *dense_bid2part) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    float result;
    int offset = tid % DENSE_BLOCK_SIZE;
    __shared__ float shm_val[TILE_SIZE];
    __shared__ int shm_idx[TILE_SIZE];

    // 计算该线程块实际对应的需要计算的位置
    int posi = dense_bid2posi[bid];
    int part = dense_bid2part[bid];
    if (posi >= num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    
    int length = min(TILE_SIZE, end - begin - part * TILE_SIZE);

    if (offset < length) {
        shm_val[offset] = val[begin + part * TILE_SIZE + offset];
        shm_idx[offset] = idx[begin + part * TILE_SIZE + offset];
    }
    __syncthreads();

    for (int j = 0; j < INFEATURE; j += DENSE_BLOCK_SIZE) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < length; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j] * shm_val[i];
        }
        atomicAdd(&vout[posi * INFEATURE + offset + j], result);
    }
}

__global__ void spmm_kernel_sparse_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE,
    int *sparse_bid2posi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int offset = tid % 32;
    float result;

    int posi = sparse_bid2posi[bid];
    if (posi > num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];

    __shared__ float shm_val[TILE_SIZE];
    __shared__ int shm_idx[TILE_SIZE];

    if (offset < end - begin) {
        shm_val[offset] = val[begin + offset];
        shm_idx[offset] = idx[begin + offset];
    }
    __syncwarp();

    for (int j = 0; j < INFEATURE; j += 32) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < end - begin; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j] * shm_val[i];
        }
        vout[posi * INFEATURE + offset + j] = result;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    // TODO: your code
    // 计算稠密行的个数和应该分配的总共的线程块数
    dense_rows = 0;
    dense_blocks_num = 0;
    sparse_blocks_num = 0;

    // 这里需要将device的数据转移到host
    int *h_ptr = new int[num_v + 1];
    int *h_idx = new int[num_e];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            dense_rows += 1;
            dense_blocks_num += (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
        }
    }
    int *h_dense_bid2posi = new int[dense_blocks_num];
    int *h_dense_bid2part = new int[dense_blocks_num];
    int *h_dense_min_idx = new int[dense_blocks_num];
    int *h_dense_max_idx = new int[dense_blocks_num];

    int *h_sparse_bid2posi = new int[num_v - dense_rows];
    int temp = 0;

    for (int i = 0, j = 0, k = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            temp = (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
            for (int p = 0; p < temp; p++) {
                h_dense_bid2posi[j+p] = i;
                h_dense_bid2part[j+p] = p;
                h_dense_min_idx[j+p] = h_idx[h_ptr[i] + p * TILE_SIZE];
                h_dense_max_idx[j+p] = h_idx[min(h_ptr[i] + (p+1) * TILE_SIZE, h_ptr[i+1])];
            }
            j += temp;
        } else {
            if (h_ptr[i+1] - h_ptr[i] != 0) {
                h_sparse_bid2posi[k] = i;
                sparse_blocks_num++;
                k++;
            }
        }
    }

    using Quadruple = std::tuple<int, int, int, int>;
    Quadruple* quadruples = new Quadruple[dense_blocks_num];
    for (int i = 0; i < dense_blocks_num; ++i) {
        quadruples[i] = std::make_tuple(h_dense_min_idx[i], h_dense_max_idx[i], h_dense_bid2posi[i], h_dense_bid2part[i]);
    }
    std::sort(quadruples, quadruples + dense_blocks_num, [](const Quadruple& a, const Quadruple& b) {
        if (std::get<0>(a) == std::get<0>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<0>(a) < std::get<0>(b);
    });
    for (int i = 0; i < dense_blocks_num; ++i) {
        h_dense_bid2posi[i] = std::get<2>(quadruples[i]);
        h_dense_bid2part[i] = std::get<3>(quadruples[i]);
    }

    // 清理动态分配的内存
    delete[] quadruples;

    checkCudaErrors(cudaMalloc2((void **)&d_dense_bid2posi, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_dense_bid2part, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_dense_min_idx, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_sparse_bid2posi, (num_v - dense_rows) * sizeof(int)));
        
    checkCudaErrors(cudaMemcpy(d_dense_bid2posi, h_dense_bid2posi, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dense_bid2part, h_dense_bid2part, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dense_min_idx, h_dense_min_idx, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sparse_bid2posi, h_sparse_bid2posi, (num_v - dense_rows) * sizeof(int), cudaMemcpyHostToDevice));

    dense_grid.x = dense_blocks_num;
    dense_block.x = DENSE_BLOCK_SIZE;

    sparse_grid.x = sparse_blocks_num;
    sparse_block.x = 32;

    delete[] h_sparse_bid2posi;
    delete[] h_ptr;
    delete[] h_idx;
}

void SpMMOpt::run(float *vin, float *vout) {
    spmm_kernel_dense_256<<<dense_grid, dense_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_dense_bid2posi, d_dense_bid2part);
    spmm_kernel_sparse_256<<<sparse_grid, sparse_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_sparse_bid2posi);
    // spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}