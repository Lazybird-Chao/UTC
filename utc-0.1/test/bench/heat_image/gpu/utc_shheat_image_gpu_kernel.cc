/*
 * utc_shheat_image_gpu_kernel.cc
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 *
 *
 *      cuda implementation of heat image program
 *      we use one cuda thread to compute one point of image.
 *      mx, my is the extended image row/column, adding one row/column to the image
 *      boundary.
 *      but we creat cuda thread based on the orignal size of image(mx-2, my-2)
 *      so, thread dim index plus 1 will be the pos in (mx, my)
 *
 *		cuda kernel do not have total grid threads synchronize in side kernel,
 *		so we iterate outside kernel program, iterative launch kernel
 *
 *		for newer cuda, 7 or 8 maybe, in kernel program, we can invoke
 *		another kernel at end of one kernel program, may be used for this
 *		iterate running.
 */

#include "utc_shheat_image_gpu_kernel.h"

__global__ void heatImage_kernel(
		double *f,
		double *newf,
		double *r,
		double rdx2,
		double rdy2,
		double beta,
		int mx,
		int my
		){

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// image row of current thread
	int j = bx * blockDim.x + tx +1; //this is column
	int i = by * blockDim.y + ty +1; //this is row
	// cuda total grid may be larger than mx, my, so only the in image
	// thread do compute
	// naive implementation, no use of shared memory
	if(i < mx-1 && j< my-1 ){
		newf[i*my + j] =
			((f[(i - 1)*my + j] + f[(i + 1)*my + j]) * rdx2 +
			 (f[i*my + j - 1] + f[i*my + j + 1]) * rdy2 - r[i*my + j]) * beta;
	}
	__syncthreads();

}


