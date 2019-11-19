
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <math.h>
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024

#include <mxnet/base.h>
#include <stdio.h>
namespace mxnet
{
namespace op
{


__global__ void forward_kernel_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K){
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K-1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = W_out/TILE_WIDTH;
    if (W_out%TILE_WIDTH != 0)
        W_grid++;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define xs2d(i1, i0) X_shared[(i1)*(X_tile_width) + (i0)]
#define ws2d(i1, i0) W_shared[(i1)* K + (i0)]
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;
    float acc = 0;
    for(int c = 0; c < C; c++) {
        if ((h0 < K) && (w0 < K)) {
            ws2d(h0, w0) = k4d(m, c, h0, w0);
        }
        __syncthreads();
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                    xs2d(i - h_base, j - w_base) = (i < H && j < W) ? x4d(n, c, i, j) : 0;
            }
        }
        __syncthreads();
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if((h0 + p) < X_tile_width && (w0 + q) < X_tile_width) acc += xs2d(h0 + p, w0 + q) * ws2d(p, q);
            }
        }
        __syncthreads();
        if(n < B && m < M && h < H_out && w < W_out) y4d(n, m, h, w) = acc;
    }
#undef y4d
#undef x4d
#undef k4d
#undef xs2d
#undef ws2d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Set the kernel dimensions
    int W_grid = ceil(W_out/(float)TILE_WIDTH);
    int H_grid = ceil(H_out/(float)TILE_WIDTH);
    int Z = H_grid*W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);
    // share mem size
    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
    // Call the kernel
    forward_kernel_shared<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    // printf("%d", W_out);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
