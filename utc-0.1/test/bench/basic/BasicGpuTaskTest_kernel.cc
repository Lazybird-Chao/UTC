/*
 * BasicGpuTaskTest_kernel.cc
 *
 *  Created on: Nov 1, 2016
 *      Author: chao
 */

#include "BasicGpuTaskTest_kernel.h"

__global__ void kernelTest(float *vec1, float* vec2, float *vec3){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int idx = bx * blockDim.x + threadIdx.x;
	vec3[idx] = vec1[idx] * vec2[idx];
	__syncthreads();
}


