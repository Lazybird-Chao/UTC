/*
 * rotate_kernel.h
 *
 *      Author: chao
 *
 * cuda implementation of image rotation.
 * For simple, we can use one cuda thread to deal with one pixel of the
 * dst-image.
 * To deal with very large image, we let one thread compute batchx*batchy
 * pixels. So the cuda block size will change from (bxsize, bysize) to
 * (bxsize*batchx, bysize*batchy)
 *
 */

#ifndef ROTATE_KERNEL_H_
#define ROTATE_KERNEL_H_


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
