/*
 * mm_kernel.h
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MM_UTC_GPU_GPU_MM_KERNEL_H_
#define BENCHAPPS_MM_UTC_GPU_GPU_MM_KERNEL_H_

template <typename T>
__global__ void gpuMatrixMulKernel(T *A,
		T *B,
		T *C,
		int matrixSizeM,
		int matrixSizeN,
		int matrixSizeP,
		int blocksize);



#endif /* BENCHAPPS_MM_UTC_GPU_GPU_MM_KERNEL_H_ */
