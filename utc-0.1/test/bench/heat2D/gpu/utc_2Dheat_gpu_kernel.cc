/*
 * utc_2Dheat_gpu_kernel.cc
 *
 *  Created on: Nov 8, 2016
 *      Author: chao
 *
 *
 *      gpu implementation of jacobi() for Heat2D test
 *      one cuda thread compute for one point
 */


#include "utc_2Dheat_gpu_kernel.h"


__global__ void jacobi_kernel(
		float *current_ptr,
		float *next_ptr,
		int my_start_row,
		int my_end_row,
		int width,
		int heigth){
	//float(*c_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])current_ptr;
	//float(*n_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])next_ptr;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int j = by*blockDim.y + ty + my_start_row; //row
	int i = bx*blockDim.x + tx;				  //column

	if(j < my_end_row && i < width){
		next_ptr[(j - my_start_row)*width + i] =
					.25 * (
					get_val_par( current_ptr, i - 1,j, width, heigth, my_start_row)
				+ get_val_par ( current_ptr, i + 1, j,width,heigth, my_start_row)
				+ get_val_par ( current_ptr, i, j - 1,width, heigth,my_start_row)
				+ get_val_par ( current_ptr,i, j + 1,width, heigth,my_start_row)
				- (pow (H, 2) * f (i, j))
				);
				enforce_bc_par (next_ptr, i, j, width, my_start_row);
	}

	__syncthreads();
}


