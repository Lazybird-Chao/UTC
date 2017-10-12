/*
 * mm_kernel.cu
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */
#include "mm_kernel.h"
#include <stdio.h>

template <typename T>
__global__ void gpuMatrixMulKernel(T *A,
							T *B,
							T *C,
							int matrixSizeM,
							int matrixSizeN,
							int matrixSizeP,
							int BLOCKSIZE){
	// Block index
	    int bx = blockIdx.x;
	    int by = blockIdx.y;

	    // Thread index
	    int tx = threadIdx.x;
	    int ty = threadIdx.y;

	    // Index of the first sub-matrix of A processed by the block
	    int aBegin = matrixSizeN * BLOCKSIZE * by;

	    // Index of the last sub-matrix of A processed by the block
	    int aEnd   = aBegin + matrixSizeN - 1;

	    // Step size used to iterate through the sub-matrices of A
	    int aStep  = BLOCKSIZE;

	    // Index of the first sub-matrix of B processed by the block
	    int bBegin = BLOCKSIZE * bx;
	    int bEnd = bBegin + matrixSizeP*(matrixSizeN-1);

	    // Step size used to iterate through the sub-matrices of B
	    int bStep  = BLOCKSIZE * matrixSizeP;

	    // Csub is used to store the element of the block sub-matrix
	    // that is computed by the thread
	    T Csub = 0;

	    // Loop over all the sub-matrices of A and B
	    // required to compute the block sub-matrix
	    for (int a = aBegin, b = bBegin;
	         a <= aEnd;
	         a += aStep, b += bStep)
	    {
	        // Declaration of the shared memory array As used to
	        // store the sub-matrix of A
	        __shared__ T As[16][16];

	        // Declaration of the shared memory array Bs used to
	        // store the sub-matrix of B
	        __shared__ T Bs[16][16];

	        // Load the matrices from device memory
	        // to shared memory; each thread loads
	        // one element of each matrix
	        if( a + tx <= aEnd &&
	        		by*BLOCKSIZE+ty < matrixSizeM){
				As[ty][tx] = A[a + matrixSizeN * ty + tx];
	        }
	        if( b + matrixSizeP * ty < bEnd &&
					bx*BLOCKSIZE+tx < matrixSizeP){
				Bs[ty][tx] = B[b + matrixSizeP * ty + tx];
	        }

	        // Synchronize to make sure the matrices are loaded
	        __syncthreads();

	        // Multiply the two matrices together;
	        // each thread computes one element
	        // of the block sub-matrix
	//#pragma unroll
	        int kend = BLOCKSIZE;
	        if(a+BLOCKSIZE-1 >aEnd)
	        	kend = matrixSizeN % BLOCKSIZE;
	        if(bx*BLOCKSIZE+tx < matrixSizeP &&
				by*BLOCKSIZE+ty < matrixSizeM){
	        for (int k = 0; k < kend; ++k)
	        {
	            Csub += As[ty][k] * Bs[k][tx];
	        }
	        }

	        // Synchronize to make sure that the preceding
	        // computation is done before loading two new
	        // sub-matrices of A and B in the next iteration
	        __syncthreads();
	    }

	    int c = matrixSizeP * BLOCKSIZE * by + BLOCKSIZE * bx;
	    if(bx*BLOCKSIZE+tx < matrixSizeP &&
			by*BLOCKSIZE+ty < matrixSizeM){
	    	C[c + matrixSizeP * ty + tx] = Csub;
	    }

}

template __global__ void gpuMatrixMulKernel(float *A,
		float *B,
		float *C,
		int matrixSizeM,
		int matrixSizeN,
		int matrixSizeP,
		int BLOCKSIZE);
template __global__ void gpuMatrixMulKernel(double *A,
		double *B,
		double *C,
		int matrixSizeM,
		int matrixSizeN,
		int matrixSizeP,
		int BLOCKSIZE);



