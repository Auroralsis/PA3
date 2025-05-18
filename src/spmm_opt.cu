#include "spmm_opt.h"

const int TILE_SIZE = 32;
const int DENSE_BLOCK_SIZE = 32;

__global__ void spmm_kernel_dense_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE, int *dense_bid2order, int *dense_order2posi, int *sum_of_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    float result;
    int offset = tid % DENSE_BLOCK_SIZE;
    __shared__ float shm_val[TILE_SIZE];
    __shared__ int shm_idx[TILE_SIZE];

    // 计算该线程块实际对应的需要计算的位置
    int order = dense_bid2order[bid];
    int posi = dense_order2posi[order];
    if (posi >= num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    
    // 计算该线程块在该行应该计算的part的位置
    int part = order == 0 ? bid : (bid - sum_of_blocks[order-1]);

    if (begin + part * TILE_SIZE + offset < end && offset < TILE_SIZE) {
        shm_val[offset] = val[begin + part * TILE_SIZE + offset];
        shm_idx[offset] = idx[begin + part * TILE_SIZE + offset];
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < INFEATURE / DENSE_BLOCK_SIZE; j++) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE && i + begin + part * TILE_SIZE < end; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j * DENSE_BLOCK_SIZE] * shm_val[i];
        }
        atomicAdd(&vout[posi * INFEATURE + offset + j * DENSE_BLOCK_SIZE], result);
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

    #pragma unroll
    for (int j = 0; j < INFEATURE / 32; j++) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < end - begin; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j * 32] * shm_val[i];
        }
        vout[posi * INFEATURE + offset + j * 32] = result;
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
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            dense_rows += 1;
            dense_blocks_num += (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
        }
    }
    int *h_dense_bid2order = new int[dense_blocks_num];
    int *h_dense_order2posi = new int[dense_rows];
    int *h_sum_of_blocks = new int[dense_rows];
    int *h_sparse_bid2posi = new int[num_v - dense_rows];
    int temp = 0;

    for (int i = 0, j = 0, k = 0, l = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            temp = (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
            for (int p = 0; p < temp; p++) {
                h_dense_bid2order[j+p] = l;
            }
            j += temp;
            h_dense_order2posi[l] = i;
            h_sum_of_blocks[l] = l == 0 ? temp : temp + h_sum_of_blocks[l-1];
            l++;
        } else {
            if (h_ptr[i+1] - h_ptr[i] != 0) {
                h_sparse_bid2posi[k] = i;
                sparse_blocks_num++;
                k++;
            }
        }
    }
    checkCudaErrors(cudaMalloc2((void **)&d_dense_bid2order, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_dense_order2posi, dense_rows * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_sum_of_blocks, dense_rows * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_sparse_bid2posi, (num_v - dense_rows) * sizeof(int)));
        
    checkCudaErrors(cudaMemcpy(d_dense_bid2order, h_dense_bid2order, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dense_order2posi, h_dense_order2posi, dense_rows * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sum_of_blocks, h_sum_of_blocks, dense_rows * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sparse_bid2posi, h_sparse_bid2posi, (num_v - dense_rows) * sizeof(int), cudaMemcpyHostToDevice));

    dense_grid.x = dense_blocks_num;
    dense_block.x = DENSE_BLOCK_SIZE;

    sparse_grid.x = sparse_blocks_num;
    sparse_block.x = 32;
}

void SpMMOpt::run(float *vin, float *vout) {
    printf("dense rows:%d", dense_rows);
    printf("dense blocks num:%d", dense_blocks_num);
    printf("sparse blocks num:%d", sparse_blocks_num);
    spmm_kernel_dense_256<<<dense_grid, dense_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_dense_bid2order, d_dense_order2posi, d_sum_of_blocks);
    spmm_kernel_sparse_256<<<sparse_grid, sparse_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_sparse_bid2posi);
    // spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}