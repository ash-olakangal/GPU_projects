#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>

#include "mma_intrinsics.cuh"

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

template<typename T>
void printResult(T *Matrix, int row_size, int column_size){

	std::cout << std::fixed << std::setprecision(4);

    	for(int i = 0; i < row_size; i++){
    	    for(int j = 0; j < column_size; j++){
    	        // Matrix[i][j] becomes:
    	        std::cout << Matrix[i * column_size + j] << " ";
    	    }
    	    std::cout << std::endl;
    	}
}

void initMatrix(std::vector<__half>& Matrix, int row_size, int column_size, int layout, std::string filename){
    	std::vector<float> numbers;
    	float temp;
    	std::ifstream file(filename);

    	if (!file.is_open()) {
    	    	std::cerr << "Could not open file: " << filename << std::endl;
    	    	return;
    	}

    	while(file >> temp){
    	    	numbers.push_back(temp);
    	}
    	
    	int k = 0; // Index for the flat vector
    	if(layout == 0){ // Row Major
    	    	for(int i = 0; i < row_size; i++){
    	    	   	for(int j = 0; j < column_size; j++){
    	    	    	    Matrix[i * column_size + j] = numbers[k++];
    	    	    	}
    	    	}
    	}
    	else {// column major
    	    for(int i = 0; i < row_size; i++){
    	        for(int j = 0; j < column_size; j++){
    	            Matrix[j * row_size + i] = numbers[i * column_size + j];
    	        }
    	    }
    	}
}

void software_pipeline(int M, int N, int K){

	//host variables
 	std::vector<__half> hA(M * K);
  	std::vector<__half> hB(K * N);  // stored column-major
  	std::vector<float> hC(M * N, 0.0f);
  	std::vector<float> hRef(M * N, 0.0f);

	//std::cout << "Initializing A: " << std::endl;
   	initMatrix(hA, M, K, 0, "A.txt");
	//std::cout << "Initializing B: " << std::endl;
   	initMatrix(hB, K, N, 1, "B.txt");

	//device variables
  	__half* dA = nullptr;
  	__half* dB = nullptr;
  	float* dC = nullptr;

  	cudaMalloc(&dA, sizeof(__half) * M * K);
  	cudaMalloc(&dB, sizeof(__half) * K * N);
  	cudaMalloc(&dC, sizeof(float) * M * N);

  	cudaMemcpy(dA, hA.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
  	cudaMemcpy(dB, hB.data(), sizeof(__half) * K * N, cudaMemcpyHostToDevice);
  	cudaMemset(dC, 0, sizeof(float) * M * N);

  	dim3 block(THREADS_PER_BLOCK, 1, 1);
  	dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M), 1);

   	int smemBytes = 65536;
  	cudaFuncSetAttribute(matrixMulKernelPipelined<PIPE_STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

  	cudaEventRecord(start);
  	matrixMulKernelPipelined<PIPE_STAGES><<<grid, block, smemBytes>>>(M, N, K, dA, dB, dC);
  	cudaEventRecord(stop);

  	cudaEventSynchronize(stop);

  	float milliseconds = 0;
  	cudaEventElapsedTime(&milliseconds, start, stop);

  	cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  	std::cout << "Execution time using software pipeline: " << milliseconds << " ms" << std::endl;

	// used for bringup and testing
	//printResult(hC.data(), M, N);

	//clean up malloc memory
  	cudaFree(dA);
  	cudaFree(dB);
  	cudaFree(dC);

}

int main(int argc, char* argv[]) {
  int M;
  int N;
  int K;
  if (argc == 4) {
    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);
  }
  else{
 	std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
 	return 1;
  }

  software_pipeline(M,N,K);

  return 0;
}
