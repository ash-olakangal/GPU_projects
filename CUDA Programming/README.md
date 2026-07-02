## Software Pipelined Tensor Core GEMM

## Overview

This project implements a software-pipelined matrix multiplication kernel in CUDA using tensor cores. Aim is to improve on the earlier Tensor Core GEMM by overlapping **global-memory data movement** with **matrix multiply-accumulate (MMA) computation** using CUDA pipeline objects

## Objective

Without software pipelining the MMA loop followed a simple sequence:

1. Load a tile of `A` and `B` from global memory to shared memory  
2. Perform Tensor Core MMA  
3. Repeat for the next `K` tile  

While correct, that approach causes Tensor Cores to stall while waiting for memory. The purpose of this assignment is to remove that bottleneck by using software pipelining, so that the next tile is fetched asynchronously while the current tile is being computed.

## Kernel Design

The implemented kernel follows the following structure:

- 4 warps per thread block
- 128 threads per block (dim3(128,1,1))
- each thread block computes one 64×64 tile of output C
- each iteration loads:
  - one 64×64 tile of A
  - one 64×64 tile of B
- computation uses the provided:
  - `mma_m16n8k16_f16_f16_smem_row_col_64x64()` primitive

Because the primitive expects exactly 4 sequential warps, the block layout is fixed to match that requirement.
- A: row-major - `fp16`
- B: column-major - `fp16`
- C: row-major - `fp32`

## Build and Run

Files:

- main.cu — for input setup, kernel launch, and timing
- launchStudentKernel.cu — for integration with the testbench
- mma_intrinsics.cuh — provided MMA wrappers
- Makefile — build targets for testing and grading

commands:

```bash
make main
make pa4_testbench
./pa4_testbench 256 256 128
make grade
```
