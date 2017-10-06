/*
 * mc_kernel.h
 *
 *  Created on: Oct 5, 2017
 *      Author: chaoliu
 */

#ifndef MC_KERNEL_H_
#define MC_KERNEL_H_

#define __MAX_RAND__ 0x7fffffff

__global__ void mc_kernel(
		double *res,
		double upper,
		double lower,
		unsigned int seed,
		long loop);




#endif /* MC_KERNEL_H_ */
