/*
 * c-ray_kernel_v2.h
 *
 *  Created on: Feb 13, 2017
 *      Author: chao
 */

#ifndef C_RAY_KERNEL_V2_H_
#define C_RAY_KERNEL_V2_H_

#include "../common.h"


/*
 * global vars for device functions
 */
extern __device__  global_vars g_vars_d;
extern __device__  vec3_t lights_d[MAX_LIGHTS];
extern __device__  vec2_t urand_d[NRAN];
extern __device__  int irand_d[NRAN];


/*
 * cuda kernels
 */
__global__ void render_kernel(
		unsigned int *pixels,
		vec3_t *pos,
		material_t *mat,
		FTYPE *rad);



#endif /* C_RAY_KERNEL_V2_H_ */
