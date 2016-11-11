/*
 * utc_matrix_gpu_kernel.h
 *
 *  Created on: Nov 5, 2016
 *      Author: chao
 */

#ifndef UTC_MATRIX_GPU_KERNEL_H_
#define UTC_MATRIX_GPU_KERNEL_H_

#define _dataType double

__global__ void gpuMatrixKernel(_dataType *A,
		_dataType *B,
		_dataType *C,
		int matrixSize,
		int blocksize);


#endif /* UTC_MATRIX_GPU_KERNEL_H_ */
