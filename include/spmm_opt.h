#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    int num_target;
    int *target, *ptr_scheduled;
    int dense_rows, dense_blocks_num;

    // 用于表示顺序对应的实际稀疏矩阵中的posi
    int *d_dense_order2posi, *d_dense_bid2order, *d_sum_of_blocks;
    // 稀疏行直接用bid对应实际稀疏矩阵中的posi
    int *d_sparse_bid2posi;

    dim3 dense_grid;
    dim3 sparse_grid;
    dim3 dense_block;
    dim3 sparse_block;
};
#endif