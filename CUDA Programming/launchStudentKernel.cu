#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>

#include "mma_intrinsics.cuh"
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// A is row major, B is column major, C is row major
// A, B and C are pointers to the matrices


namespace cg = cooperative_groups;

const int TILE_M = 64;
const int TILE_N = 64;
const int TILE_K = 64;
const int THREADS_PER_BLOCK = 128;
const int PIPE_STAGES = 2;

__host__ __device__ constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

template <int STAGES>
__global__ void matrixMulKernelPipelined(int M, int N, int K, const __half* A, const __half* B, float* C) {
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pipeline_state;
	
	extern __shared__ __align__(128) unsigned char smem_raw[];
	__half* A_stage[STAGES];
	__half* B_stage[STAGES];
	
	auto block = cg::this_thread_block();
	auto pipe = cuda::make_pipeline(block, &pipeline_state);
	
	int tid = threadIdx.x;
	int blockRow = blockIdx.y * TILE_M;
	int blockCol = blockIdx.x * TILE_N;
	int kTiles = K / TILE_K;

	size_t offset = 0;

	for (int s = 0; s < STAGES; ++s) {
		A_stage[s] = reinterpret_cast<__half*>(smem_raw + offset);
	  	offset += TILE_M * TILE_K * sizeof(__half);
	}
	for (int s = 0; s < STAGES; ++s) {
	  	B_stage[s] = reinterpret_cast<__half*>(smem_raw + offset);
	  	offset += TILE_K * TILE_N * sizeof(__half);
	}
	float* C_shared = reinterpret_cast<float*>(smem_raw + offset);
	
	
	for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
	  	C_shared[idx] = 0.0f;
	}
	block.sync();
	
	auto async_copy_stage = [&](int stage, int kBase) {
	  // A tile row-major 64 x 64
	  for (int chunk = tid; chunk < TILE_M * (TILE_K / 8); chunk += THREADS_PER_BLOCK) {
	    	int row = chunk / (TILE_K / 8);
	    	int col8 = (chunk % (TILE_K / 8)) * 8;
	    	const __half* srcA = A + (blockRow + row) * K + (kBase + col8);
	    	__half* dstA = A_stage[stage] + row * TILE_K + col8;
		// calling mnemcpy_async
	    	cuda::memcpy_async(dstA, srcA, cuda::aligned_size_t<16>(sizeof(__half) * 8), pipe);
	  }
	
	  //B tile: column-major 64 x 64
	  for (int chunk = tid; chunk < TILE_N * (TILE_K / 8); chunk += THREADS_PER_BLOCK) {
	    	int col = chunk / (TILE_K / 8);
	    	int row8 = (chunk % (TILE_K / 8)) * 8;
	    	const __half* srcB = B + (blockCol + col) * K + (kBase + row8);
	    	__half* dstB = B_stage[stage] + col * TILE_K + row8;
		// calling mnemcpy_async
	      cuda::memcpy_async(dstB, srcB, cuda::aligned_size_t<16>(sizeof(__half) * 8), pipe);
	  }
	};
	
	pipe.producer_acquire();
	async_copy_stage(0, 0);
	pipe.producer_commit();
	
	for (int tile = 1; tile < kTiles; ++tile) {
	  const int compute_stage = (tile - 1) % STAGES;
	  const int copy_stage = tile % STAGES;
	
	  pipe.producer_acquire();
	  async_copy_stage(copy_stage, tile * TILE_K);
	  pipe.producer_commit();
	
	  pipe.consumer_wait();
	  block.sync();
	  mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[compute_stage], B_stage[compute_stage], C_shared);
	  block.sync();
	  pipe.consumer_release();
	}
	
	pipe.consumer_wait();
	block.sync();
	mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[(kTiles - 1) % STAGES], B_stage[(kTiles - 1) % STAGES], C_shared);
	block.sync();
	pipe.consumer_release();
	
	for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
	  int row = idx / TILE_N;
	  int col = idx % TILE_N;
	  int globalRow = blockRow + row;
	  int globalCol = blockCol + col;
	  if (globalRow < M && globalCol < N) {
	    C[globalRow * N + globalCol] = C_shared[idx];
	  }
	}
}


void launchStudentKernel(int M, int N, int K, __half* A,
                                              __half* B,
                                              float* C) {
  // Launch your kernel here with appropriate grid and block sizes...
  // Uncomment code below to increase shared memory size


  	dim3 block(THREADS_PER_BLOCK, 1, 1);
  	dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M), 1);

   	int smemBytes = 65536;
  	cudaFuncSetAttribute(matrixMulKernelPipelined<PIPE_STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

  	matrixMulKernelPipelined<PIPE_STAGES><<<grid, block, smemBytes>>>(M, N, K, A, B, C);
}
