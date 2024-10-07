#include "kernel.h"
#include "cuda_runtime.h"
#include "unistd.h"
#include "iostream"

// # define BLOCK_TILE_SIZE_X 32
// # define BLOCK_TILE_SIZE_Y 32
// # define BLOCK_TILE_SIZE_K 32
// # define BLOCK_TILE_SKEW_SIZE_K 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

using std::cout;

template <size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>

__device__ void load_data_from_global_memory_to_shared_memory(
    float const* A, size_t lda, float const* B, size_t ldb,
    float A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    float B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
    #pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};


        float val{static_cast<float>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);

        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
    // Load data from B(global mem) on DRAM to tile on shared memory.
    #pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                        NUM_THREADS;    ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        float val{static_cast<float>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);

        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
};


template < size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, float const* A,
                         size_t lda, float const* B, size_t ldb, float beta, float* C,
                         size_t ldc)
{
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ float A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ float B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    float sum{static_cast<float>(0)};
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_from_global_memory_to_shared_memory<
            BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

        #pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

__global__ void GEMM_naiev(const float *A, const float *B, float *C, 
    int M, int N, int K) {
    // tx<=N, ty<=M
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(ty < M && tx < N) {
        float c = 0;
        for(int i = 0; i < K; ++i){
            printf("%f , %f \n",A[ty * K + i] ,   B[i * N + tx]);
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}


void linear(array2d_t<float>& X, array2d_t<float>& W, array2d_t<float>& output1){;
    int M= X.row_count;
    int K= X.col_count;
    int N= W.col_count;
    // int thr_per_blk = 256;
    // int blk_in_grid=ceil(float(N) / thr_per_blk);
    // dim3 BlockDim(32, 32);  
    // // cudaMemset( X.data_ptr , 1,4*sizeof(float));
    // dim3 GridDim(2, 2);  
    // // cudaMemset(output1.data_ptr , 4*sizeof(float) , 2 );

    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(N) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(M) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v02< 32U, 32U, 32U><<<grid_dim, block_dim, 1024, 0 >>>(M, N, K, X.data_ptr, K, W.data_ptr, N, 1, output1.data_ptr, N);    
    // GEMM_naiev<<<1, GridDim>>>(X.data_ptr, W.data_ptr, output1.data_ptr, M, N, K);
    // sgemm<<<1,1>>>(M,N,K,X.data_ptr, W.data_ptr,output1.data_ptr);
    // cout<<cudaGetErrorName(cudaGetLastError());
    // cudaDeviceSynchronize();
}

