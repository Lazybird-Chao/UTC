/*
 * yuv.cc
 *
 *  Created on: Nov 13, 2017
 *      Author: Chao
 */
#include "common.h"

#include <stdint.h>
#include <cmath>

void yuv(int iter, int w, int h, uint8_t *r, uint8_t *g, uint8_t *b,
		uint8_t **y, uint8_t **u, uint8_t **v){
	int iterations = iter;
	*y = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	*u = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	*v = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	uint8_t R,G,B,Y,U,V;
	for(int i=0; i<iterations; i++){
		for(int j=0; j< w*h; j++){
			R = r[j];
			G = g[j];
			B = b[j];
			Y = (uint8_t)round(0.256788*R+0.504129*G+0.097906*B) + 16;
			U = (uint8_t)round(-0.148223*R-0.290993*G+0.439216*B) + 128;
			V = (uint8_t)round(0.439216*R-0.367788*G-0.071427*B) + 128;
			(*y)[j] = Y;
			(*u)[j] = U;
			(*v)[j] = V;
		}
	}
}


