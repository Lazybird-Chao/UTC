/*
 * rotate_kernel.cc
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 */

#include "rotate_kernel.h"



__device__ inline double myround(double num, int digits) {
    double v[] = {1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
	if(digits > (sizeof(v)/sizeof(double))) return num;
    return floor(num * v[digits] + 0.5) / v[digits];
}

__device__ inline void interpolateLinear(Pixel* a, Pixel* b, Pixel* dest, float weight) {
	dest->r = a->r * (1.0-weight) + b->r * weight;
	dest->g = a->g * (1.0-weight) + b->g * weight;
	dest->b = a->b * (1.0-weight) + b->b * weight;
}

__device__ inline void filter(Pixel* colors, Pixel* dest, float2* sample_pos){
	Pixel sample_v_upper, sample_v_lower;
	float x_weight = myround(sample_pos->x - floor(sample_pos->x), PRECISION);
	float y_weight = myround(sample_pos->y - floor(sample_pos->y), PRECISION);

	interpolateLinear(&colors[0], &colors[3], &sample_v_upper, x_weight);
	interpolateLinear(&colors[1], &colors[2], &sample_v_lower, x_weight);
	interpolateLinear(&sample_v_upper, &sample_v_lower, dest, y_weight);
}

__global__ void rotate_kernel(
		Pixel *inImg,
		int inW,
		int inH,
		Pixel *outImg,
		int outW,
		int outH,
		int angle,
		int start_row,
		int end_row,
		int batchx,
		int batchy){

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float x_offset_src = (float)inW/2.0;
	float y_offset_src = (float)inH/2.0;
	float x_offset_dst = (float)outW/2.0;
	float y_offset_dst = (float)outH/2.0;

	for(int i=0; i<batchy; i++){
		int idy = by*blockDim.y*batchy + ty + i*blockDim.y + start_row;
		for(int j=0; j<batchx; j++){
			int idx = bx*blockDim.x*batchx + tx + j*blockDim.x;
			if(idx<outW && idy<=end_row){
				int rev_angle = 360-angle;
				float2 cur = {-x_offset_dst+(float)idx, y_offset_dst-(float)idy};
				float2 origin_pix;
				rotatePoint(cur, origin_pix,rev_angle);
				/* original image contains this point */
				if(origin_pix.x< x_offset_src && origin_pix.x > (0.0-x_offset_src) &&
						origin_pix.y<y_offset_src && origin_pix.y>(0.0-y_offset_src)){
					int samples[4][2];
					Pixel colors[4];
					/* Get sample positions */
					for(int k = 0; k < 4; k++) {
						samples[k][0] = (int)(origin_pix.x + (float)inW/2.0) + ((k == 2 || k == 3) ? 1 : 0);
						samples[k][1] = (int)(-origin_pix.y + (float)inH/2.0) + ((k == 1 || k == 3) ? 1 : 0);
						// Must make sure sample positions are still valid image pixels
						if(samples[k][0] >= inW)
							samples[k][0] = inW-1;
						if(samples[k][1] >= inH)
							samples[k][1] = inH-1;
					}
					/* Get colors for samples */
					for(int k=0; k<4; k++){
						colors[k] = inImg[samples[k][1]*inW + samples[k][0]] ;
					}
					/* Filter colors */
					Pixel final;
					filter(colors, &final, &origin_pix);
					/* Write output */
					outImg[(idy-start_row)*outW + idx] = final;
				}
				else{
					Pixel final ={0,0,0};
					outImg[(idy-start_row)*outW + idx] = final;
				}
			}// end for one pixel
		}// end row batch
	}// end column batch

	return;
}






