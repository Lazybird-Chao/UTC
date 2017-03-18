/*
 * yuvconvert_kernel.h
 *
 *      Author: chao
 */

#ifndef YUVCONVERT_KERNEL_H_
#define YUVCONVERT_KERNEL_H_


#include "image.h"



__global__ void convert(Pixel *inImg,
		yuv_color_t outImg,
		int h,
		int w,
		int batchx,
		int batchy);



#endif /* YUVCONVERT_KERNEL_H_ */
