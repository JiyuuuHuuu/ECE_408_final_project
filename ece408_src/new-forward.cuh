
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <math.h>
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024

#include <mxnet/base.h>
#include <stdio.h>
//#include "cublasLt.h"
namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = W_out/TILE_WIDTH;
    if (W_out%TILE_WIDTH != 0)
        W_grid++;

    // int b = blockIdx.z;
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z/W_grid*TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z%W_grid*TILE_WIDTH + threadIdx.x;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if (h < H_out && w < W_out)
    {
//        for (int b = 0; b < B; b++)
//        {
            float acc = 0;
            for (int c = 0; c < C; c++)
            {
                for (int p = 0; p < K; p++)
                {
                    for (int q = 0; q < K; q++)
                    {
                        if (h + p < H && w + q < W)
                            acc += x4d(b, c, h + p, w + q)*k4d(m, c, p, q);
                    }
                }
            }
            y4d(b, m, h, w) = acc;
//        }
    }


#undef y4d
#undef x4d
#undef k4d
}


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
//__global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll){
//    int c, s, h_out, w_out, h_unroll, w_base, p, q;
//    int t = blockId.x * CUDA_MAX_NUM_THREADS + threadId.x;
//    int H_out = H - K + 1;
//    int W_out = W - K + 1;
//    int W_unroll = H_out * W_out;
//#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//    if(t < C * W_unroll){
//        c = t / W_unroll;
//        s = t % W_unroll;
//        h_out = s / W_out;
//        w_out = s % W_out;
//        h_unroll = h_out * W_out + w_out;
//        w_base = c * K * K;
//        for(p = 0; p < K; p++){
//            for(q = 0; q < K; q++){
//                w_unroll = w_base + p * K + q;
//                X_unroll[h_unroll * W_unroll + w_unroll] = x4d(c, h_out + p, w_out + q);
//            }
//        }
//    }
//#undef x4d
//}

void unroll(int C, int H, int W, int K, int b, float* X, float* X_unroll){
    int c, h, w, p, q, w_base, w_unroll, h_unroll;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    #define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    for(c = 0; c < C; c++) {
        w_base = c * (K*K);
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++) {
                for (h = 0; h < H_out; h++) {
                    for (w = 0; w < W_out; w++) {
                        w_unroll = w_base + p * K + q;
                        h_unroll = h * W_out + w;
                        X_unroll[h_unroll * C * K * K + w_unroll] = x4d(b, c, h + p, w + q);
                    }
                }
            }
        }
    }
    #undef x4d
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

//    int W_unroll = C * K * K;
//    int H_unroll = H_out * W_out;
//    float* X_unrolled = (float*) malloc(W_unroll * H_unroll * sizeof(float));
//    for (int b=0; b < B; b++) {
//        unroll(C, H, W, K, b, x, X_unrolled);
////        gemm(H_unroll, M, W_unroll, X_unrolled, W, Y[n]);
//    }
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
