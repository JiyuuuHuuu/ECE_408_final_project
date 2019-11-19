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