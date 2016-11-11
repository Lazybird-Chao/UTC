/*
 * utc_2Dheat_gpu_kernel.h
 *
 *  Created on: Nov 8, 2016
 *      Author: chao
 */

#ifndef UTC_2DHEAT_GPU_KERNEL_H_
#define UTC_2DHEAT_GPU_KERNEL_H_

#define H 1.0
#define T_SRC0 550.0

__global__ void jacobi_kernel(
		float *current_ptr,
		float *next_ptr,
		int __numProcesses,
		int __processId,
		int my_start_row,
		int my_end_row);

__device__ __host__ inline void enforce_bc_par(
		float *domain_ptr,
		//int rank,
		int i,
		int j,
		int width,
		int my_start_row
		){
	/* enforce bc's first */
	if (i == ((int) floor (width / H / 2) - 1) && j == 0) {
		/* This is the heat source location */
		domain_ptr[j* ((int) floor (width / H)) +i] = T_SRC0;
	}
	else if (i <= 0 || j <= 0 || i >= ((int) floor (width / H) - 1)
			 || j >= ((int) floor (width / H) - 1)) {
		/* All edges and beyond are set to 0.0 */
		//domain_ptr[global_to_local (rank, j)*((int) floor (width / H)) +i] = 0.0;
		domain_ptr[(j-my_start_row)*((int) floor (width / H)) +i] = 0.0;
	}
}

__device__ __host__ inline float get_val_par(
		//float *above_ptr,
		float *domain_ptr,
		//float *below_ptr,
		int i,
		int j,
		int width,
		int heigth,
		int my_start_row){
	float ret_val;

	/* enforce bc's first */
	if (i == ((int) floor (width / H / 2) - 1) && j == 0) {
		/* This is the heat source location */
		ret_val = T_SRC0;
	}
	else if (i <= 0 || j <= 0 || i >= ((int) floor (width / H) - 1)
			 || j >= ((int) floor (heigth / H) - 1)) {
		/* All edges and beyond are set to 0.0 */
		ret_val = 0.0;
	}
	else{
		/* else, return the value in the domain asked for */
		ret_val = domain_ptr[(j-my_start_row)*((int) floor (width / H)) + i];
	}
	return ret_val;
}

__device__ __host__ inline float f(int i, int j){
	return 0.0;
}

#endif /* UTC_2DHEAT_GPU_KERNEL_H_ */
