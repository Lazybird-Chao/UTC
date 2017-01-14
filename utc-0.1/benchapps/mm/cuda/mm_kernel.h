#ifndef MM_KERNEL_H_
#define MM_KERNEL_H_

#define _dataType float

__global__ void gpuMatrixKernel(_dataType *A,
		_dataType *B,
		_dataType *C,
		int matrixSize,
		int blocksize);

#endif

