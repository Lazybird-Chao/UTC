/*
 * hc_kernel.cc
 *
 */

#include "hc_kernel.h"


__global__ void jacobi_kernel(
		FTYPE *current_ptr,
		FTYPE *next_ptr,
		int height,
		int width){
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int j = by*blockDim.y + ty; //row
	int i = bx*blockDim.x + tx;				  //column

	if(j < height && i < width){
		next_ptr[j*width + i] =
					.25 * (
					get_var_par( current_ptr, i - 1,j, width, height)
				+ get_var_par ( current_ptr, i + 1, j,width,height)
				+ get_var_par ( current_ptr, i, j - 1,width, height)
				+ get_var_par ( current_ptr,i, j + 1,width, height)
				- (pow (H, 2) * f (i, j))
				);
				enforce_bc_par (next_ptr, i, j, width, height);
	}

	__syncthreads();
}

__global__ void get_convergence_sqd_kernel(
		FTYPE *current_ptr,
		FTYPE *next_ptr,
		FTYPE *converge_sqd,
		int h,
		int w){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int column = bx*blockDim.x + tx;
	FTYPE sum=0.0;
	if(column < w){
		for(int i=0; i<h; i++){
			//sum += pow((current_ptr[i*w + column] - next_ptr[i*w + column]), 2);
			sum += (current_ptr[i*w + column] - next_ptr[i*w + column])*
					(current_ptr[i*w + column] - next_ptr[i*w + column]);
		}
		converge_sqd[column] = sum;
	}
	__syncthreads();
}


