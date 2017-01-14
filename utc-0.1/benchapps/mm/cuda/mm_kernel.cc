#include "mm_kernel.h"

__global__ void gpuMatrixKernel(_dataType *A,
							_dataType *B,
							_dataType *C,
							int matrixSize, int BLOCK_SIZE){
	// Block index
	    int bx = blockIdx.x;
	    int by = blockIdx.y;

	    // Thread index
	    int tx = threadIdx.x;
	    int ty = threadIdx.y;

	    // Index of the first sub-matrix of A processed by the block
	    int aBegin = matrixSize * BLOCK_SIZE * by;

	    // Index of the last sub-matrix of A processed by the block
	    int aEnd   = aBegin + matrixSize - 1;

	    // Step size used to iterate through the sub-matrices of A
	    int aStep  = BLOCK_SIZE;

	    // Index of the first sub-matrix of B processed by the block
	    int bBegin = BLOCK_SIZE * bx;

	    // Step size used to iterate through the sub-matrices of B
	    int bStep  = BLOCK_SIZE * matrixSize;

	    // Csub is used to store the element of the block sub-matrix
	    // that is computed by the thread
	    _dataType Csub = 0;

	    // Loop over all the sub-matrices of A and B
	    // required to compute the block sub-matrix
	    for (int a = aBegin, b = bBegin;
	         a <= aEnd;
	         a += aStep, b += bStep)
	    {

	        // Declaration of the shared memory array As used to
	        // store the sub-matrix of A
	        __shared__ _dataType As[32][32];

	        // Declaration of the shared memory array Bs used to
	        // store the sub-matrix of B
	        __shared__ _dataType Bs[32][32];

	        // Load the matrices from device memory
	        // to shared memory; each thread loads
	        // one element of each matrix
	        As[ty][tx] = A[a + matrixSize * ty + tx];
	        Bs[ty][tx] = B[b + matrixSize * ty + tx];

	        // Synchronize to make sure the matrices are loaded
	        __syncthreads();

	        // Multiply the two matrices together;
	        // each thread computes one element
	        // of the block sub-matrix
	#pragma unroll

	        for (int k = 0; k < BLOCK_SIZE; ++k)
	        {
	            Csub += As[ty][k] * Bs[k][tx];
	        }

	        // Synchronize to make sure that the preceding
	        // computation is done before loading two new
	        // sub-matrices of A and B in the next iteration
	        __syncthreads();
	    }

	    int c = matrixSize * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	    C[c + matrixSize * ty + tx] = Csub;
}
