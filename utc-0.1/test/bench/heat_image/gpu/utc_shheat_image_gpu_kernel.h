/*
 * utc_shheat_image_gpu_kernel.h
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 */

#ifndef UTC_SHHEAT_IMAGE_GPU_KERNEL_H_
#define UTC_SHHEAT_IMAGE_GPU_KERNEL_H_


__global__ void heatImage_kernel(
		double *f,
		double *newf,
		double *r,
		double rdx2,
		double rdy2,
		double beta,
		int mx,
		int my
		);



#endif /* UTC_SHHEAT_IMAGE_GPU_KERNEL_H_ */
