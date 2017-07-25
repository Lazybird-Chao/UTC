/*
 * md5_kernel.h
 *
 *  Created on: Feb 26, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MD5_CUDA_MD5_KERNEL_H_
#define BENCHAPPS_MD5_CUDA_MD5_KERNEL_H_

#include <stdint.h>

typedef unsigned int MD5_u32plus;
typedef struct __attribute__((aligned(4))){
	MD5_u32plus lo, hi;
	MD5_u32plus a, b, c, d;
	unsigned char buffer[64];
	//MD5_u32plus block[16];
} MD5_CTX;


__global__ void md5_process(
		uint8_t *inputs,
		uint8_t *out,
		long numbuffs,
		long buffsize);

__device__ void MD5_Init(MD5_CTX *ctx);

__device__ void MD5_Update(MD5_CTX *ctx, void *data, long offset, unsigned long size);

__device__ void MD5_Final(unsigned char *result, MD5_CTX *ctx, int offset);

__device__ void *body_with_global(MD5_CTX *ctx, void *data, long offset, unsigned long size);

__device__ void *body_with_local(MD5_CTX *ctx, void *data, unsigned long size);

#endif /* BENCHAPPS_MD5_CUDA_MD5_KERNEL_H_ */
