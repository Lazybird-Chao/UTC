/*
 * rotate_kernel.h
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 */

#ifndef ROTATE_KERNEL_H_
#define ROTATE_KERNEL_H_

#include "../image.h"

#define PI 3.14159
#define PRECISION 3

__global__ void rotate_kernel(
		Pixel *inImg,
		int inW,
		int inH,
		Pixel *outImg,
		int outW,
		int outH,
		int angle,
		int batchx,
		int batchy);

__device__ __host__ inline void rotatePoint(float2 &pt, float2 &target, int angle){
	float rad = (float)angle/180 * PI;
	target.x = pt.x * cos(rad) - pt.y * sin(rad);
	target.y = pt.x * sin(rad) + pt.y * cos(rad);
}



#endif /* ROTATE_KERNEL_H_ */
