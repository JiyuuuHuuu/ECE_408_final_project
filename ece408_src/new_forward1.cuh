
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

//        __global__ void unroll_Kernel(int K, float* W, float* W_unroll){
//            int c = blockIdx.x;
//            int
//        }

        __global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll){
            int c, s, h_out, w_out, h_unroll, w_base, p, q;
            int t = blockId.x * CUDA_MAX_NUM_THREADS + threadId.x;
            int b = blockIdx.y;
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            int W_unroll = H_out * W_out;
#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            if(t < C * W_unroll){
                c = t / W_unroll;
                s = t % W_unroll;
                h_out = s / W_out;
                w_out = s % W_out;
                h_unroll = h_out * W_out + w_out;
                w_base = c * K * K;
                for(p = 0; p < K; p++){
                    for(q = 0; q < K; q++){
                        w_unroll = w_base + p * K + q;
                        if (h_out + p < H && w_out + q < W && h_unroll < K * K * C && w_unroll < W_unroll)
                            X_unroll[b * (K * K * C * W_unroll) + h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
                        else
                            X_unroll[b * (K * K * C * W_unroll) + h_unroll * W_unroll + w_unroll] = 0.0;
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
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            float* W_unroll;
            cudaMalloc(&(W_unroll), sizeof(float) * K * K * C * M);
            for (int m = 0; m < M; ++m){
                for (int c = 0; c < C; ++c){
                    for (int p = 0; p < K; ++p){
                        for (int q = 0; q < K; ++q){
                            int k = p*K + q;
                            W_unroll[m * K * K * C + K * K * c + k] = k4d(m,c,p,q);
                        }
                    }
                    // cudaMemcpy( &(W_unroll[m * K * K * C + K * K * c]), &(k4d(m,c,0,0)), K * K * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }

            float* X_unroll;
            cudaMalloc(&(X_unroll), sizeof(float) * H_out * W_out * C * K * K);
            dim3 blockDim(CUDA_MAX_NUM_THREADS, 1, 1); // FIXME: get cuda_max_num_thread from device query
            dim3 gridDim(ceil(C * H_out * W_out / (float) CUDA_MAX_NUM_THREADS), B, 1);
            unroll_Kernel<<<gridDim, blockDim>>>(C, H, W, K, x.dptr_, X_unroll);
            // Set the kernel dimensions
//            int W_grid = ceil(W_out/(float)TILE_WIDTH);
//            int H_grid = ceil(H_out/(float)TILE_WIDTH);
//            int Z = H_grid*W_grid;
//
//            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//            dim3 gridDim(B, M, Z);
//            // share mem size
//            size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
            // Call the kernel
//            forward_kernel_shared<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
            // printf("%d", W_out);
            // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
            MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
#undef k4d

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
