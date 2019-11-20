
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
        __global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll){
            int c, s, h_out, w_out, h_unroll, w_base, p, q, w_unroll;
            int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
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

        __global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int numBatch) {
          //@@ Insert code to implement matrix multiplication here
          //@@ You have to use shared memory for this MP
          __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
          __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

          int bx = blockIdx.x;
          int by = blockIdx.y;
          int tx = threadIdx.x;
          int ty = threadIdx.y;

          int x = bx*blockDim.x + tx;
          int y = by*blockDim.y + ty;
          float ret = 0.0;

          for (int b = 0; b < numBatch; b++){
              for (int i = 0; i < ceil(numAColumns/(float)TILE_WIDTH); i++){
                if(x < numCColumns && i*blockDim.y + ty < numAColumns){
                  tileB[ty][tx] = B[b*numBRows*numBColumns + (i*blockDim.y + ty)*numBColumns + x];
                }
                else{
                  tileB[ty][tx] = 0;
                }
                if(y < numCRows && i*blockDim.x + tx < numAColumns){
                  tileA[ty][tx] = A[y*numAColumns + i*blockDim.x + tx];
                }
                else{
                  tileA[ty][tx] = 0;
                }
                __syncthreads();

                if(x < numCColumns && y < numCRows){
                  for(int q = 0; q < blockDim.x; q++){
                    ret += tileA[ty][q]*tileB[q][tx];
                  }
                }
                __syncthreads();
              }
              if(x < numCColumns && y < numCRows)
                C[b*numCRows*numCColumns + y*numCColumns + x] = ret;
          }
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
#define k4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            float* W_unroll;
            cudaMalloc(&(W_unroll), sizeof(float) * K * K * C * M);
            for (int m = 0; m < M; ++m){
                for (int c = 0; c < C; ++c){
                    cudaMemcpy( &(W_unroll[m * K * K * C + K * K * c]), &(w.dptr_[(m) * (C * K * K) + (c) * (K * K)]), K * K * sizeof(float), cudaMemcpyDeviceToDevice); // FIXME: error: taking address of temporary
                }
            }
            float* X_unroll;
            cudaMalloc(&(X_unroll), sizeof(float) * H_out * W_out * C * K * K);
            dim3 blockDim(CUDA_MAX_NUM_THREADS, 1, 1); // FIXME: get cuda_max_num_thread from device query
            dim3 gridDim(ceil(C * H_out * W_out / (float) CUDA_MAX_NUM_THREADS), B, 1);
            unroll_Kernel<<<gridDim, blockDim>>>(C, H, W, K, x.dptr_, X_unroll);
            MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

            // matrix multiplication
            int numARows = M;
            int numAColumns = K*K*C;
            int numBRows = K*K*C;
            int numBColumns = H_out*W_out;
            int numCRows = numARows;
            int numCColumns = numBColumns;
            float* deviceC;
            cudaMalloc((void**) &deviceC, B*numCRows*numCColumns*sizeof(float));
            //float* HostUnroll = (float *)malloc(numARows*numBColumns*sizeof(float));
            dim3 DimGrid(ceil(numCColumns/(float)TILE_WIDTH), ceil(numCRows/(float)TILE_WIDTH), 1);
            dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
            matrixMultiplyShared<<<DimGrid, DimBlock>>>(W_unroll, X_unroll, deviceC,
                                        numARows, numAColumns, numBRows,
                                        numBColumns, numCRows, numCColumns, B);

            cudaMemcpy(y.dptr_, deviceC, B*numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);
            // Set the kernel dimensions
//            int W_grid = ceil(W_out/(float)TILE_WIDTH);
//            int H_grid = ceil(H_out/(float)TILE_WIDTH);
//            int Z = H_grid*W_grid;
//
//            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//            dim3 gridDim(B, M, Z);
            // Call the kernel
//            forward_kernel_shared<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
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
