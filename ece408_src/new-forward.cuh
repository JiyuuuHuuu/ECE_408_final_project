/* unroll and matrix multiplication optimization */
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <math.h>
#define TILE_WIDTH_L1 16
#define TILE_WIDTH_L2 16
#define CHANNEL_TILE 4
#define BATCH_CLUSTER_LENGTH 4

#include <mxnet/base.h>
#include <stdio.h>

namespace mxnet
{
    namespace op
    {
        // Layer 1 no parallel
        __global__ void matrixMultiplyShared_L1(float * __restrict__ A, float * __restrict__ B, float * __restrict__ Out,
                                                int numARows, int numAColumns,
                                                int numBRows, int numBColumns,
                                                int numCRows, int numCColumns,
                                                int K, int C, int W, int H, int W_out) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
#define x4d(i3, i2, i1, i0) B[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            __shared__ float subTileA[TILE_WIDTH_L1][TILE_WIDTH_L1];
            __shared__ float subTileB[TILE_WIDTH_L1][TILE_WIDTH_L1];
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * TILE_WIDTH_L1 + ty;
            int Col = bx * TILE_WIDTH_L1 + tx;

            int b = blockIdx.z;
            float Pval = 0;
#pragma unroll
            for (int m = 0; m < ceil(25/(TILE_WIDTH_L1*1.0)); ++m){
                int temp_row = m * TILE_WIDTH_L1 + ty;
                subTileA[ty][tx] = (Row < numCRows  && m*TILE_WIDTH_L1+tx < numAColumns) ? A[Row*numAColumns + m*TILE_WIDTH_L1+tx] : 0;
                // implicit unrolling
                int X_b = b;
                int X_c = temp_row/(K*K);
                int X_p = (temp_row%(K*K))/K, X_q = (temp_row%(K*K))%K;
                int X_h = Col/W_out, X_w = Col%W_out;
                subTileB[ty][tx] = (Col < numBColumns && temp_row < numBRows) ? x4d(X_b, X_c, X_h + X_p, X_w + X_q) : 0;
                __syncthreads();
                if(Row < numCRows && Col < numCColumns){
#pragma unroll
                    for (int k = 0; k < TILE_WIDTH_L1; k++){
                        Pval += subTileA[ty][k] * subTileB[k][tx];
                    }
                }
                __syncthreads();
                if(Row < numCRows && Col < numCColumns) Out[b * numCColumns * numCRows + Row*numCColumns+Col] = Pval;
            }
#undef x4d
        }
        // Layer 2 no parallel
        __global__ void matrixMultiplyShared_L2(float * __restrict__ A, float * __restrict__ B, float * __restrict__ Out,
                                                int numARows, int numAColumns,
                                                int numBRows, int numBColumns,
                                                int numCRows, int numCColumns,
                                                int K, int C, int W, int H, int W_out) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
#define x4d(i3, i2, i1, i0) B[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) A[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            __shared__ float subTileA[CHANNEL_TILE][TILE_WIDTH_L2][TILE_WIDTH_L2];
            __shared__ float subTileB[CHANNEL_TILE][TILE_WIDTH_L2][TILE_WIDTH_L2];
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tz = threadIdx.z;
            int Row = by * TILE_WIDTH_L2 + ty;
            int Col = bx * TILE_WIDTH_L2 + tx;

            int bz = blockIdx.z / CHANNEL_TILE;
            int b = blockIdx.z / CHANNEL_TILE; // % 4, / 4
            int Channel = tz % 4 + bz * CHANNEL_TILE;
            float Pval = 0;
#pragma unroll
            for (int m = 0; m < ceil(25/(TILE_WIDTH_L2*1.0)); ++m){ // m < ceil(25/12*(TILE_WIDTH_L2*1.0))
                int temp_row = m * TILE_WIDTH_L2 + ty;
                subTileA[tz][ty][tx] = (Row < numCRows  && m*TILE_WIDTH_L2+tx < numAColumns && Channel < C) ? k4d(Row, Channel, ((m*TILE_WIDTH_L2+tx)%Channel)/K, ((m*TILE_WIDTH_L2+tx)%Channel)%K) : 0;
                // implicit unrolling
                int X_b = b;
                int X_c = Channel;
                int X_p = (temp_row%(Channel))/K, X_q = (temp_row%(Channel))%K;
                int X_h = Col/W_out, X_w = Col%W_out;
                subTileB[tz][ty][tx] = (Col < numBColumns && temp_row < numBRows && Channel < C) ? x4d(X_b, X_c, X_h + X_p, X_w + X_q) : 0;
                __syncthreads();
                if(Row < numCRows && Col < numCColumns){
#pragma unroll
                    for (int k = 0; k < TILE_WIDTH_L2; k++){
                        Pval += subTileA[tz][ty][k] * subTileB[tz][k][tx];
                    }
                }
                __syncthreads();
                if(Row < numCRows && Col < numCColumns) atomicAdd(&Out[b * numCColumns * numCRows + Row*numCColumns+Col], Pval);
            }
#undef x4d
        }
        // Layer 1 parallel
        __global__ void matrixMultiplyShared_parallel_L1(float * __restrict__ A, float * __restrict__ B, float * __restrict__ Out,
                                                         int numARows, int numAColumns,
                                                         int numBRows, int numBColumns,
                                                         int numCRows, int numCColumns,
                                                         int K, int C, int W, int H, int W_out, int numBatch) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
#define x4d(i3, i2, i1, i0) B[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            __shared__ float subTileA[TILE_WIDTH_L1][TILE_WIDTH_L1];
            __shared__ float subTileB[BATCH_CLUSTER_LENGTH][TILE_WIDTH_L1][TILE_WIDTH_L1];
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * TILE_WIDTH_L1 + ty;
            int Col = bx * TILE_WIDTH_L1 + tx;

            int b = threadIdx.z;
            int b_idx = blockIdx.z * BATCH_CLUSTER_LENGTH + threadIdx.z;
            float Pval = 0;
#pragma unroll
            for (int m = 0; m < ceil(25/(TILE_WIDTH_L1*1.0)); ++m){
                int temp_row = m * TILE_WIDTH_L1 + ty;
                subTileA[ty][tx] = (Row < numCRows  && m*TILE_WIDTH_L1+tx < numAColumns) ? A[Row*numAColumns + m*TILE_WIDTH_L1+tx] : 0;
                // implicit unrolling
                int X_b = b_idx;
                int X_c = temp_row/(K*K);
                int X_p = (temp_row%(K*K))/K, X_q = (temp_row%(K*K))%K;
                int X_h = Col/W_out, X_w = Col%W_out;
                subTileB[b][ty][tx] = (Col < numBColumns && temp_row < numBRows && X_b < numBatch) ? x4d(X_b, X_c, X_h + X_p, X_w + X_q) : 0;
                __syncthreads();
                if(Row < numCRows && Col < numCColumns){
#pragma unroll
                    for (int k = 0; k < TILE_WIDTH_L1; k++){
                        Pval += subTileA[ty][k] * subTileB[b][k][tx];
                    }
                }
                __syncthreads();
                if(Row < numCRows && Col < numCColumns) Out[b_idx * numCColumns * numCRows + Row*numCColumns+Col] = Pval;
            }
#undef x4d
        }
        // Layer 2 parallel
        __global__ void matrixMultiplyShared_parallel_L2(float * __restrict__ A, float * __restrict__ B, float * __restrict__ Out,
                                                         int numARows, int numAColumns,
                                                         int numBRows, int numBColumns,
                                                         int numCRows, int numCColumns,
                                                         int K, int C, int W, int H, int W_out, int numBatch) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
#define x4d(i3, i2, i1, i0) B[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            __shared__ float subTileA[TILE_WIDTH_L2][TILE_WIDTH_L2];
            __shared__ float subTileB[BATCH_CLUSTER_LENGTH][TILE_WIDTH_L2][TILE_WIDTH_L2];
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * TILE_WIDTH_L2 + ty;
            int Col = bx * TILE_WIDTH_L2 + tx;

            int b = threadIdx.z;
            int b_idx = blockIdx.z * BATCH_CLUSTER_LENGTH + threadIdx.z;
            float Pval = 0;
#pragma unroll
            for (int m = 0; m < ceil(25*12/(TILE_WIDTH_L2*1.0)); ++m){
                int temp_row = m * TILE_WIDTH_L2 + ty;
                subTileA[ty][tx] = (Row < numCRows  && m*TILE_WIDTH_L2+tx < numAColumns) ? A[Row*numAColumns + m*TILE_WIDTH_L2+tx] : 0;
                // implicit unrolling
                int X_b = b_idx;
                int X_c = temp_row/(K*K);
                int X_p = (temp_row%(K*K))/K, X_q = (temp_row%(K*K))%K;
                int X_h = Col/W_out, X_w = Col%W_out;
                subTileB[b][ty][tx] = (Col < numBColumns && temp_row < numBRows && X_b < numBatch) ? x4d(X_b, X_c, X_h + X_p, X_w + X_q) : 0;
                __syncthreads();
                if(Row < numCRows && Col < numCColumns){
#pragma unroll
                    for (int k = 0; k < TILE_WIDTH_L2; k++){
                        Pval += subTileA[ty][k] * subTileB[b][k][tx];
                    }
                }
                __syncthreads();
                if(Row < numCRows && Col < numCColumns) Out[b_idx * numCColumns * numCRows + Row*numCColumns+Col] = Pval;
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
//            printf("B = %d, M = %d, C = %d, H = %d, W = %d, K = %d\n", B, M, C, H, W, K);

            // matrix multiplication
            int numARows = M;
            int numAColumns = K*K*C;
            int numBRows = K*K*C;
            int numBColumns = H_out*W_out;
            int numCRows = numARows;
            int numCColumns = numBColumns;
            if( M / 24 == 0){
                // Layer 1
                dim3 DimGrid(ceil(numCColumns/(float)TILE_WIDTH_L1), ceil(numCRows/(float)TILE_WIDTH_L1), ceil(B/(float)BATCH_CLUSTER_LENGTH));
                dim3 DimBlock(TILE_WIDTH_L1, TILE_WIDTH_L1, BATCH_CLUSTER_LENGTH);
                matrixMultiplyShared_parallel_L1<<<DimGrid, DimBlock>>>(w.dptr_, x.dptr_, y.dptr_,
                        numARows, numAColumns, numBRows,
                        numBColumns, numCRows, numCColumns,
                        K, C, H, W, W_out, B);
            }else{
                // Layer 2
                dim3 DimGrid(ceil(numCColumns/(float)TILE_WIDTH_L2), ceil(numCRows/(float)TILE_WIDTH_L2), B * ceil(C/(float)CHANNEL_TILE));
                dim3 DimBlock(TILE_WIDTH_L2, TILE_WIDTH_L2, CHANNEL_TILE);
                cudaMemset(y.dptr_, 0, sizeof(float)* M * H_out * W_out);
                matrixMultiplyShared_L2<<<DimGrid, DimBlock>>>(w.dptr_, x.dptr_, y.dptr_,
                        numARows, numAColumns, numBRows,
                        numBColumns, numCRows, numCColumns,
                        K, C, H, W, W_out);
            }

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
