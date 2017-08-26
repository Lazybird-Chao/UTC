/*
 * micro_kernel.h
 *
 *
 */

#ifndef BENCHAPPS_MICRO_UTC_MICRO_KERNEL_H_
#define BENCHAPPS_MICRO_UTC_MICRO_KERNEL_H_


__global__ void micro_kernel(
		float* data,
		int offset,
		int nscale,
		int loop);
__global__ void micro_kernel(
		double* data,
		int offset,
		int nscale,
		int loop);


#endif /* BENCHAPPS_MICRO_UTC_MICRO_KERNEL_H_ */
