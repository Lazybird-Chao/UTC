/*
 * yuvconvert_kernel.cc
 *
 *      Author: chao
 */

#include "yuvconvert_kernel.h"


__global__ void convert(Pixel *inImg,
		uint8_t* img_y,
		uint8_t* img_u,
		uint8_t* img_v,
		int h,
		int w,
		int batchx,
		int batchy){
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	uint8_t *pY = img_y;
	uint8_t *pU = img_u;
	uint8_t *pV = img_v;

	for(int i=0; i<batchy; i++){
		int idy = by*blockDim.y*batchy + ty + i*blockDim.y;
		for(int j=0; j<batchx; j++){
			int idx = bx*blockDim.x*batchx + tx + j*blockDim.x;
			if(idx<w && idy<h){
				uint8_t R,G,B,Y,U,V;
				R = inImg[idy*w+idx].r;
				G = inImg[idy*w+idx].g;
				B = inImg[idy*w+idx].b;
				Y = (uint8_t)round(0.256788*R+0.504129*G+0.097906*B) + 16;
				U = (uint8_t)round(-0.148223*R-0.290993*G+0.439216*B) + 128;
				V = (uint8_t)round(0.439216*R-0.367788*G-0.071427*B) + 128;
				pY[idy*w+idx] = Y;
				pU[idy*w+idx] = U;
				pV[idy*w+idx] = V;
			}
		}
	}
	return;
}

