#include "spmm_opt.h"

const int WARP_SIZE = 32;

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    // ptr, idx, val分别是稀疏矩阵的CSR格式对应的数组
    // vin, vout分别是与稀疏矩阵相乘的稠密矩阵和最终结果的稠密矩阵
    // num_v是稀疏矩阵的行数，即M*M中的M
    // INFEATURE是输入的稠密矩阵的列数，M*K中的K

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // 对于INFEARTURE = 32 或 256，分别确定一行计算需要使用的线程块个数
    int lines_num = INFEATURE / WARP_SIZE;

    // 根据线程id求出该线程负责的具体稀疏矩阵中的一行以及其需要计算的列
    int row_of_thr = tid / (WARP_SIZE * lines_num);
    int line_of_thr = tid % (WARP_SIZE * lines_num);

    if (row_of_thr >= num_v) return;
    int begin = ptr[row_of_thr], end = ptr[row_of_thr + 1];

    float result = 0.0f;
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
}

void SpMMOpt::run(float *vin, float *vout) {
    // TODO: your code
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}