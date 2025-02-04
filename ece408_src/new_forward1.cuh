/* unroll and matrix multiplication optimization */
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <math.h>
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 64

#include <mxnet/base.h>
#include <stdio.h>

namespace mxnet
{
    namespace op
    {
        __global__ void unroll_Kernel(int C, int H, int W, int K, int B, float* X, float* X_unroll){
            int c, s, h_out, w_out, h_unroll, w_base, p, q, w_unroll;
            int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
            int b = blockIdx.y;
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            int W_unroll = H_out * W_out;
#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            if(t < C * W_unroll && b < B){
                c = t / W_unroll;
                s = t % W_unroll;

                h_out = s / W_out;
                w_out = s % W_out;

                w_unroll = h_out * W_out + w_out;
                w_base = c * K * K;
                for(p = 0; p < K; p++){
                    for(q = 0; q < K; q++){
                        h_unroll = w_base + p * K + q;
//                        if (h_out + p < H && w_out + q < W && h_unroll < K * K * C && w_unroll < W_unroll) // Fixme: unnecessary ?
                        X_unroll[b * (K * K * C * W_unroll) + h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
//                        else
//                            X_unroll[b * (K * K * C * W_unroll) + h_unroll * W_unroll + w_unroll] = 0.0;  // Fixme: unnecessary ?
                    }
                }
            }
#undef x4d
        }
        __global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                             int numARows, int numAColumns,
                                             int numBRows, int numBColumns,
                                             int numCRows, int numCColumns) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
            __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
            __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;

            int b = blockIdx.z;
            float Pval = 0;
            for (int m = 0; m < ceil(numAColumns/(TILE_WIDTH*1.0)); ++m){
                subTileA[ty][tx] = (Row < numCRows    && m*TILE_WIDTH+tx < numAColumns) ? A[Row*numAColumns + m*TILE_WIDTH+tx] : 0;
                subTileB[ty][tx] = (Col < numCColumns && m*TILE_WIDTH+ty < numBRows) ? B[b * numBColumns * numBRows + (m*TILE_WIDTH+ty)*numBColumns+Col] : 0;
                __syncthreads();
                if(Row < numCRows && Col < numCColumns){
                    for (int k = 0; k < TILE_WIDTH; k++){
                        Pval += subTileA[ty][k] * subTileB[k][tx];
                    }
                }
                __syncthreads();
                if(Row < numCRows && Col < numCColumns) C[b * numCColumns * numCRows + Row*numCColumns+Col] = Pval;
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
//            printf("B = %d, M = %d, C = %d, H = %d, W = %d, K = %d\n", B, M, C, H, W, K);
            size_t freeMem, totalMem;
            cudaSetDevice(0);
            cudaMemGetInfo(&freeMem, &totalMem);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
//            printf("total = %zu\n", deviceProp.totalGlobalMem);
//            printf("need = %zu\n", (size_t)(sizeof(float) * H_out * W_out * C * K * K * B));
//            printf("total = %zu\n", totalMem);
//            printf("free = %zu\n", freeMem);
#define k4d_dptr(i3, i2, i1, i0) w.dptr_[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            if (freeMem >= (size_t)(sizeof(float) * H_out * W_out * C * K * K * B)){
                float* X_unroll;
                cudaMalloc(&(X_unroll), sizeof(float) * H_out * W_out * C * K * K * B);
                dim3 blockDim(CUDA_MAX_NUM_THREADS, 1, 1); // FIXME: get cuda_max_num_thread from device query
                dim3 gridDim(ceil(C * H_out * W_out / (float) CUDA_MAX_NUM_THREADS), B, 1);
                unroll_Kernel<<<gridDim, blockDim>>>(C, H, W, K, B, x.dptr_, X_unroll);
                MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

                // matrix multiplication
                int numARows = M;
                int numAColumns = K*K*C;
                int numBRows = K*K*C;
                int numBColumns = H_out*W_out;
                int numCRows = numARows;
                int numCColumns = numBColumns;

                dim3 DimGrid(ceil(numCColumns/(float)TILE_WIDTH), ceil(numCRows/(float)TILE_WIDTH), B);
                dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
                matrixMultiplyShared<<<DimGrid, DimBlock>>>(w.dptr_, X_unroll, y.dptr_,
                        numARows, numAColumns, numBRows,
                        numBColumns, numCRows, numCColumns);

                MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
                cudaFree(X_unroll);
            } else{
                int B_prime = floor(0.7*(float)freeMem/(float)(sizeof(float) * H_out * W_out * C * K * K));
//                printf("B_prime = %d\n", B_prime);
                int B_left = B;
                float* curr_y = y.dptr_;
                float* curr_x = x.dptr_;
                while (B_left > 0){
                    int B_temp = B_left>=B_prime?B_prime:B_left;
//                    printf("B_left = %d\n", B_left);
                    B_left -= B_temp;
//                    printf("B_temp = %d\n", B_temp);
                    float* X_unroll;
                    cudaError_t err = cudaMalloc(&(X_unroll), sizeof(float) * H_out * W_out * C * K * K * B_temp);
//                    if (err == cudaErrorMemoryAllocation)
//                        printf("Buuu  ");
//                    if (err == cudaSuccess)
//                        printf("Yay  ");
//                    printf("mallocsize = %zu\n", (size_t)(sizeof(float) * H_out * W_out * C * K * K * B_temp));
                    dim3 blockDim(CUDA_MAX_NUM_THREADS, 1, 1); // FIXME: get cuda_max_num_thread from device query
                    dim3 gridDim(ceil(C * H_out * W_out / (float) CUDA_MAX_NUM_THREADS), B_temp, 1);
                    unroll_Kernel<<<gridDim, blockDim>>>(C, H, W, K, B_temp, curr_x, X_unroll);
                    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
                    // curr_x += H_out * W_out * C * K * K * B_temp;
                    curr_x = &curr_x[C * H * W * B_temp];
                    // printf("curr_x: %f", *curr_x);

                    // matrix multiplication
                    int numARows = M;
                    int numAColumns = K*K*C;
                    int numBRows = K*K*C;
                    int numBColumns = H_out*W_out;
                    int numCRows = numARows;
                    int numCColumns = numBColumns;

                    dim3 DimGrid(ceil(numCColumns/(float)TILE_WIDTH), ceil(numCRows/(float)TILE_WIDTH), B_temp);
                    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
                    matrixMultiplyShared<<<DimGrid, DimBlock>>>(w.dptr_, X_unroll, curr_y,
                            numARows, numAColumns, numBRows,
                            numBColumns, numCRows, numCColumns);
                    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
                    curr_y = &curr_y[B_temp*numCRows*numCColumns];
                    cudaFree(X_unroll);
                }
            }
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
