#include "spmm_opt.h"

constexpr int TILE_SIZE_32 = 64;
constexpr int TILE_SIZE_256 = 32;
constexpr int BLOCK_SIZE = 32;

void mergeRowEntries(int* h_ptr, int* h_idx, float* h_val, int num_v) {
    std::vector<int> new_ptr = {0};
    std::vector<int> new_idx;
    std::vector<float> new_val;

    for (int i = 0; i < num_v; ++i) {
        std::unordered_map<int, float> index_map;

        for (int j = h_ptr[i]; j < h_ptr[i + 1]; ++j) {
            int idx = h_idx[j];
            float val = h_val[j];
            index_map[idx] += val;
        }

        for (auto& entry : index_map) {
            new_idx.push_back(entry.first);
            new_val.push_back(entry.second);
        }
        new_ptr.push_back(static_cast<int>(new_idx.size()));
    }
    std::copy(new_ptr.begin(), new_ptr.end(), h_ptr);
    std::copy(new_idx.begin(), new_idx.end(), h_idx);
    std::copy(new_val.begin(), new_val.end(), h_val);
}

__global__ void spmm_kernel_dense_32(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE, int *dense_bid2posi, int *dense_bid2part) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    float result;
    int offset = tid % BLOCK_SIZE;
    __shared__ float shm_val[TILE_SIZE_32];
    __shared__ int shm_idx[TILE_SIZE_32];

    // 计算该线程块实际对应的需要计算的位置
    int posi = dense_bid2posi[bid];
    int part = dense_bid2part[bid];
    if (posi >= num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    
    int length = min(TILE_SIZE_32, end - begin - part * TILE_SIZE_32);

    for (int i = 0; i < TILE_SIZE_32 / BLOCK_SIZE && offset + i * BLOCK_SIZE < length; i++) {
        shm_val[offset + i * BLOCK_SIZE] = val[begin + part * TILE_SIZE_32 + offset + i * BLOCK_SIZE];
        shm_idx[offset + i * BLOCK_SIZE] = idx[begin + part * TILE_SIZE_32 + offset + i * BLOCK_SIZE];
    }
    __syncthreads();

    for (int j = 0; j < INFEATURE; j += BLOCK_SIZE) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < length; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j] * shm_val[i];
        }
        atomicAdd(&vout[posi * INFEATURE + offset + j], result);
    }
}

__global__ void spmm_kernel_dense_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE, int *dense_bid2posi, int *dense_bid2part) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    float result;
    int offset = tid % BLOCK_SIZE;
    __shared__ float shm_val[TILE_SIZE_256];
    __shared__ int shm_idx[TILE_SIZE_256];

    // 计算该线程块实际对应的需要计算的位置
    int posi = dense_bid2posi[bid];
    int part = dense_bid2part[bid];
    if (posi >= num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    
    int length = min(TILE_SIZE_256, end - begin - part * TILE_SIZE_256);

    for (int i = 0; i < TILE_SIZE_256 / BLOCK_SIZE && offset + i * BLOCK_SIZE < length; i++) {
        shm_val[offset + i * BLOCK_SIZE] = val[begin + part * TILE_SIZE_256 + offset + i * BLOCK_SIZE];
        shm_idx[offset + i * BLOCK_SIZE] = idx[begin + part * TILE_SIZE_256 + offset + i * BLOCK_SIZE];
    }
    __syncthreads();

    for (int j = 0; j < INFEATURE; j += BLOCK_SIZE) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < length; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j] * shm_val[i];
        }
        atomicAdd(&vout[posi * INFEATURE + offset + j], result);
    }
}

__global__ void spmm_kernel_sparse_32(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE,
    int *sparse_bid2posi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int offset = tid % BLOCK_SIZE;
    float result;

    int posi = sparse_bid2posi[bid];
    if (posi > num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];

    __shared__ float shm_val[TILE_SIZE_32];
    __shared__ int shm_idx[TILE_SIZE_32];

    for (int i = 0; i < TILE_SIZE_32 / BLOCK_SIZE && offset + i * BLOCK_SIZE < end - begin; i++) {
        shm_val[offset + i * BLOCK_SIZE] = val[begin + offset + i * BLOCK_SIZE];
        shm_idx[offset + i * BLOCK_SIZE] = idx[begin + offset + i * BLOCK_SIZE];
    }
    __syncthreads();

    for (int j = 0; j < INFEATURE; j += BLOCK_SIZE) {
        result = 0.0f;
        #pragma unroll
        for (int i = 0; i < end - begin; i++) {
            result += vin[shm_idx[i] * INFEATURE + offset + j] * shm_val[i];
        }
        vout[posi * INFEATURE + offset + j] = result;
    }
}

__global__ void spmm_kernel_sparse_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE,
    int *sparse_bid2posi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int offset = tid % BLOCK_SIZE;
    float result;

    int posi = sparse_bid2posi[bid];
    if (posi > num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];

    __shared__ float shm_val[TILE_SIZE_256];
    __shared__ int shm_idx[TILE_SIZE_256];

    for (int i = 0; i < TILE_SIZE_256 / BLOCK_SIZE && offset + i * BLOCK_SIZE < end - begin; i++) {
        shm_val[offset + i * BLOCK_SIZE] = val[begin + offset + i * BLOCK_SIZE];
        shm_idx[offset + i * BLOCK_SIZE] = idx[begin + offset + i * BLOCK_SIZE];
    }
    __syncthreads();

    for (int j = 0; j < INFEATURE; j += BLOCK_SIZE) {
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
    int TILE_SIZE = feat_in == 32 ? TILE_SIZE_32 : TILE_SIZE_256;
    // 计算稠密行的个数和应该分配的总共的线程块数
    dense_rows = 0;
    dense_blocks_num = 0;
    // 这里的spare_blocks_num除去了所有稠密行和空行，因此不能直接减
    sparse_blocks_num = 0;

    // 这里需要将device的数据转移到host
    int *h_ptr = new int[num_v + 1];
    int *h_idx = new int[num_e];
    float *h_val = new float[num_e];

    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost));

    mergeRowEntries(h_ptr, h_idx, h_val, num_v);

    for (int i = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            dense_rows += 1;
            dense_blocks_num += (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
        }
    }
    int *h_dense_bid2posi = new int[dense_blocks_num];
    int *h_dense_bid2part = new int[dense_blocks_num];
    int *h_dense_min_idx = new int[dense_blocks_num];

    int *h_sparse_bid2posi = new int[num_v - dense_rows];
    int temp = 0;

    for (int i = 0, j = 0, k = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] > TILE_SIZE) {
            temp = (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
            for (int p = 0; p < temp; p++) {
                h_dense_bid2posi[j+p] = i;
                h_dense_bid2part[j+p] = p;
                h_dense_min_idx[j+p] = h_idx[h_ptr[i] + p * TILE_SIZE];
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

    using Triple = std::pair<int, std::pair<int, int>>;
    Triple* triples = new Triple[dense_blocks_num];

    for (int i = 0; i < dense_blocks_num; ++i) {
        triples[i] = std::make_pair(h_dense_min_idx[i], std::make_pair(h_dense_bid2posi[i], h_dense_bid2part[i]));
    }
    std::sort(triples, triples + dense_blocks_num, [](const Triple& a, const Triple& b) {
        return a.first < b.first;
    });
    for (int i = 0; i < dense_blocks_num; ++i) {
        h_dense_min_idx[i] = triples[i].first;
        h_dense_bid2posi[i] = triples[i].second.first;
        h_dense_bid2part[i] = triples[i].second.second;
    }

    checkCudaErrors(cudaMalloc2((void **)&d_dense_bid2posi, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_dense_bid2part, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_dense_min_idx, dense_blocks_num * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void **)&d_sparse_bid2posi, sparse_blocks_num * sizeof(int)));
        
    checkCudaErrors(cudaMemcpy(d_dense_bid2posi, h_dense_bid2posi, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dense_bid2part, h_dense_bid2part, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dense_min_idx, h_dense_min_idx, dense_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sparse_bid2posi, h_sparse_bid2posi, sparse_blocks_num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idx, h_idx, num_e * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, h_val, num_e * sizeof(float), cudaMemcpyHostToDevice));

    dense_grid.x = dense_blocks_num;
    dense_block.x = BLOCK_SIZE;

    sparse_grid.x = sparse_blocks_num;
    sparse_block.x = BLOCK_SIZE;

    delete[] h_sparse_bid2posi;
    delete[] h_dense_bid2part;
    delete[] h_dense_min_idx;
    delete[] h_ptr;
    delete[] h_idx;
    delete[] triples;
}

void SpMMOpt::run(float *vin, float *vout) {
    if (feat_in == 32) {
        spmm_kernel_dense_32<<<dense_grid, dense_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
            d_dense_bid2posi, d_dense_bid2part);
        spmm_kernel_sparse_32<<<sparse_grid, sparse_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
            d_sparse_bid2posi);
    } else {
        spmm_kernel_dense_256<<<dense_grid, dense_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
            d_dense_bid2posi, d_dense_bid2part);
        spmm_kernel_sparse_256<<<sparse_grid, sparse_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
            d_sparse_bid2posi);
    }
}