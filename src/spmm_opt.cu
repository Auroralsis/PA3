#include "spmm_opt.h"

const int WARP_SIZE = 32;
const int TILE_SIZE = 256;

__global__ void print(int dense_rows, int dense_blocks_num, int *d_dense_order2posi, int *d_dense_bid2order) {
    printf("dense_order2posi\n");
    for (int i = 0; i < dense_rows; i++) {
        printf("%d ",d_dense_order2posi[i]);
    }
    printf("dense_bid2order\n");
    for (int i = 0; i < dense_blocks_num; i++) {
        printf("%d ",d_dense_bid2order[i]);
    }
}

__global__ void spmm_kernel_dense_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE, int *dense_bid2order, int *dense_order2posi, int *sum_of_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int offset = tid % INFEATURE;

    // 计算该线程块实际对应的需要计算的位置
    int order = dense_bid2order[bid];
    int posi = dense_order2posi[order];
    if (posi > num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    
    // 计算该线程块在该行应该计算的part的位置
    int part = order == 0 ? bid : (bid - sum_of_blocks[order-1]);

    if (part != 0) return;
    float result = 0.0f;
    #pragma unroll
    for (int i = begin; i < end; i++) {
        result += vin[idx[i] * INFEATURE + offset] * val[i];
    }
    vout[posi * INFEATURE + offset] = result;
    // int length = (part + 1) * TILE_SIZE > (end - begin) ? (end - begin) % TILE_SIZE : TILE_SIZE;

    // float result = 0.0f;
    // #pragma unroll
    // for (int i = begin + part * TILE_SIZE; i < length + begin + part * TILE_SIZE; i++) {
    //     result += vin[idx[i] * INFEATURE + offset] * val[i];
    // }
    // atomicAdd(&vout[posi * INFEATURE + offset], result);
}

__global__ void spmm_kernel_sparse_256(int *ptr, int *idx, float *val, float *vin, float *vout,int num_v, int INFEATURE,
    int *sparse_bid2posi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int offset = tid % INFEATURE;

    int posi = sparse_bid2posi[bid];
    if (posi >= num_v) return;
    int begin = ptr[posi], end = ptr[posi + 1];
    float result = 0.0f;

    #pragma unroll
    for (int i = begin; i < end; i++) {
        result += vin[idx[i] * INFEATURE + offset] * val[i];
    }
    vout[posi * INFEATURE + offset] = result;
}

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    // ptr, idx, val分别是稀疏矩阵的CSR格式对应的数组
    // vin, vout分别是与稀疏矩阵相乘的稠密矩阵和最终结果的稠密矩阵
    // num_v是稀疏矩阵的行数，即M*M中的M
    // INFEATURE是输入的稠密矩阵的列数，M*K中的K

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 对于INFEARTURE = 32 或 256，分别确定一行计算需要使用的线程块个数
    int lines_num = INFEATURE / WARP_SIZE;

    // 根据线程id求出该线程负责的具体稀疏矩阵中的一行以及其需要计算的列
    int row_of_thr = tid / (WARP_SIZE * lines_num);
    int line_of_thr = tid % (WARP_SIZE * lines_num);

    if (row_of_thr >= num_v) return;
    int begin = ptr[row_of_thr], end = ptr[row_of_thr + 1];

    float result = 0.0f;

    #pragma unroll
    for (int i = begin; i < end; i++) {
        result += vin[idx[i] * INFEATURE + line_of_thr] * val[i];
    }
    vout[row_of_thr * INFEATURE + line_of_thr] = result;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    // TODO: your code
    int ROW_SIZE = feat_in / 32;
    int BLOCK_SIZE = WARP_SIZE;
    grid.x = num_v * ROW_SIZE;
    block.x = BLOCK_SIZE;

    // 计算稠密行的个数和应该分配的总共的线程块数
    dense_rows = 0;
    dense_blocks_num = 0;

    // 这里需要将device的数据转移到host
    int *h_ptr = new int[num_v + 1];
    int *h_idx = new int[num_e];
    float *h_val = new float[num_e];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_v; i++) {
        if (h_ptr[i+1] - h_ptr[i] >= TILE_SIZE) {
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
        if (h_ptr[i+1] - h_ptr[i] >= TILE_SIZE) {
            temp = (h_ptr[i+1] - h_ptr[i] - 1) / TILE_SIZE + 1;
            for (int p = 0; p < temp; p++) {
                h_dense_bid2order[j+p] = l;
            }
            j += temp;
            h_dense_order2posi[l] = i;
            h_sum_of_blocks[l] = temp;
            l++;
        } else {
            h_sparse_bid2posi[k] = i;
            k++;
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

    // 对于稠密行的计算使用spmm_kernel_dense，每一稠密行，使用多个8*32的线程块来计算，根据该稠密行的稠密元素的数量决定
    // 稀疏行类似
    dense_grid.x = dense_blocks_num;
    dense_block.x = 8*32;

    sparse_grid.x = num_v - dense_rows;
    sparse_block.x = 8*32;
}

void SpMMOpt::run(float *vin, float *vout) {
    // TODO: your code
    dim3 grid1, block1;
    grid1.x = 1;
    block1.x = 1;
    print<<<grid1, block1>>>(dense_rows, dense_blocks_num, d_dense_order2posi, d_dense_bid2order);
    spmm_kernel_dense_256<<<dense_grid, dense_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_dense_bid2order, d_dense_order2posi, d_sum_of_blocks);
    spmm_kernel_sparse_256<<<sparse_grid, sparse_block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in,
        d_sparse_bid2posi);
    // spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}