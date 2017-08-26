#ifndef MM_KERNEL_H_
#define MM_KERNEL_H_


template <typename T>
__global__ void gpuMatrixKernel(T *A,
		T *B,
		T *C,
		int matrixSizeM,
		int matrixSizeN,
		int matrixSizeP,
		int blocksize);

#endif

