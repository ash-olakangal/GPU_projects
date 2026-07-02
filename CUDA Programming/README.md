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
- 128 threads per block (`dim3(128,1,1)`)
- each thread block computes one 64×64 tile of output `C`
- each `K` iteration loads:
  - one 64×64 tile of A
  - one 64×64 tile of B
- computation uses the provided:
  - `mma_m16n8k16_f16_f16_smem_row_col_64x64()` primitive

Because the primitive expects exactly 4 sequential warps, the block layout is fixed to match that requirement.

## Memory Layout

The kernel uses the following matrix layouts:

- `A`: **row-major**, `fp16`
- `B`: **column-major**, `fp16`
- `C`: **row-major**, `fp32`

## How the Kernel Works

### 1. Thread Block Tiling
Each thread block is responsible for one `64×64` output tile of `C`.

### 2. Shared Memory Buffers
Dynamic shared memory is partitioned into:

- `A_stage[2]` — two shared-memory buffers for `A`
- `B_stage[2]` — two shared-memory buffers for `B`
- `C_shared` — one shared-memory accumulation tile for `C`

This creates a buffer for loading the next tile while computing the current one.

### 3. Pipeline Preload
Before entering the main loop, the first `A` and `B` tiles are prefetched into shared memory using `cuda::memcpy_async()` and committed to the pipeline.

### 4. Overlapped Copy + Compute
For each subsequent `K` tile:

- the next `A` and `B` tiles are asynchronously copied into the alternate stage
- the block waits for the previously loaded stage to become ready
- Tensor Core MMA is performed on the current stage
- the stage is released and the pipeline advances

This overlaps memory movement with Tensor Core execution.

### 5. Final Drain
After the loop finishes, the final prefetched stage is consumed and accumulated into `C_shared`.

### 6. Writeback
The completed `64×64` `fp32` tile is written back to global memory in row-major format.

## Build and Run

Files:

- `main.cu` — for input setup, kernel launch, and timing
- `launchStudentKernel.cu` — for integration with the testbench
- `mma_intrinsics.cuh` — provided MMA wrappers
- `Makefile` — build targets for testing and grading

commands:

```bash
make main
make pa4_testbench
./pa4_testbench 256 256 128
make grade
