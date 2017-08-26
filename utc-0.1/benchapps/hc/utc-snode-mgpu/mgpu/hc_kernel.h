/*
 * hc_kernel.h
 *
 *
 */

#ifndef HC_KERNEL_H_
#define HC_KERNEL_H_

#include "../typeconfig.h"

#define H 1.0
#define T_SRC0 550.0


__global__ void jacobi_kernel(
		FTYPE *current_ptr,
		FTYPE *next_ptr,
		int h,
		int w,
		FTYPE *top_row,
		FTYPE *bottom_row,
		int startRowIndex,
		int total_rows);

__global__ void get_convergence_sqd_kernel(
		FTYPE *current_ptr,
		FTYPE *next_ptr,
		FTYPE *converge_sqd,
		int h,
		int w);

__device__  inline void enforce_bc_par(
		FTYPE *domain_ptr,
		int i,
		int j,
		int w,
		int h,
		int startRowIndex,
		int total_rows){
	if(i==(w/2-1) && j+startRowIndex==0){
		domain_ptr[j*w + i] = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		domain_ptr[j*w + i] = 0.0;
	}
}

__device__  inline FTYPE get_var_par(
		FTYPE *domain_ptr,
		int i, //column
		int j, //row
		int w,
		int h,
		FTYPE *top_row,
		FTYPE *bottom_row,
		int startRowIndex,
		int total_rows){

	FTYPE ret_val;
	if(i == w/2-1 && j+startRowIndex==0){
		ret_val = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		ret_val = 0.0;
	}
	else if(j<0 )
		ret_val = top_row[i];
	else if(j>h-1)
		ret_val = bottom_row[i];
	else
		ret_val = domain_ptr[j*w + i];

	return ret_val;
}

__device__  inline FTYPE f(int i, int j){
	return 0.0;
}

#endif /* HC_KERNEL_H_ */
