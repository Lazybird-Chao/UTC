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
		/*FTYPE v1, v2, v3, v4;
		//
		if(i-1 == width/2-1 && j==0){
			v1 = T_SRC0;
		}
		else if(i-1<=0 || j<=0 || i-1>=width-1 || j>=height-1){
			v1 = 0.0;
		}
		else
			v1 = current_ptr[j*width + i-1];
		//
		if(i+1 == width/2-1 && j==0){
			v2 = T_SRC0;
		}
		else if(i+1<=0 || j<=0 || i+1>=width-1 || j>=height-1){
			v2 = 0.0;
		}
		else
			v2 = current_ptr[j*width + i+1];
		//
		if(i == width/2-1 && j-1==0){
			v3 = T_SRC0;
		}
		else if(i<=0 || j-1<=0 || i>=width-1 || j-1>=height-1){
			v3 = 0.0;
		}
		else
			v3 = current_ptr[(j-1)*width + i];
		//
		if(i == width/2-1 && j+1==0){
			v4 = T_SRC0;
		}
		else if(i<=0 || j+1<=0 || i>=width-1 || j+1>=height-1){
			v4 = 0.0;
		}
		else
			v4 = current_ptr[(j+1)*width + i];
		next_ptr[j*width +i] = 0.25*(v1 + v2 + v3 + v4 - pow (H, 2) * f (i, j));
		*/


		next_ptr[j*width + i] =
					.25 * (
					get_var_par( current_ptr, i - 1,j, width, height)
				+ get_var_par ( current_ptr, i + 1, j,width,height)
				+ get_var_par ( current_ptr, i, j - 1,width, height)
				+ get_var_par ( current_ptr,i, j + 1,width, height)
				- (pow (H, 2) * f (i, j))
				);
		enforce_bc_par (next_ptr, i, j, width, height);
		/*if(i==(width/2-1) && j==0){
			next_ptr[j*width + i] = T_SRC0;
		}
		else if(i<=0 || j<=0 || i>=width-1 || j>=height-1){
			next_ptr[j*width + i] = 0.0;
		}*/
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


