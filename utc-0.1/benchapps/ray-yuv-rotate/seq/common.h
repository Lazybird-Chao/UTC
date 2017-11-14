/*
 * common.h
 *
 *  Created on: Nov 13, 2017
 *      Author: Chao
 */

#include "image.h"

#include <stdint.h>

#ifndef BENCHAPPS_RAY_YUV_ROTATE_SEQ_COMMON_H_
#define BENCHAPPS_RAY_YUV_ROTATE_SEQ_COMMON_H_

#define FTYPE float

/*
 * required data structure
 */
typedef struct vec3{
	FTYPE x,y,z;
}	vec3_t;

typedef struct ray{
	struct vec3 orig, dir;
}	ray_t;

typedef struct material {
	struct vec3 col;	/* color */
	FTYPE spow;		/* specular power */
	FTYPE refl;		/* reflection intensity */
}	material_t;

typedef struct sphere {
	struct vec3 pos;
	FTYPE rad;
	struct material mat;
	struct sphere *next;
}	sphere_t;

typedef struct spoint {
	struct vec3 pos, normal, vref;	/* position, normal and view reflection */
	FTYPE dist;		/* parametric distance of intersection along the ray */
}	spoint_t;

typedef struct camera {
	struct vec3 pos, targ;
	FTYPE fov;
}	camera_t;

typedef struct sphere2 {
	struct vec3 pos;
	FTYPE rad;
	struct material mat;
}	sphere2_t;

/*
 * sub routines declaration
 */
void render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples);
vec3_t	trace(ray_t ray, int depth);
vec3_t	shade(sphere2_t *obj, spoint_t *sp, int depth);
inline vec3_t	reflect(vec3_t &v, vec3_t &n);
inline vec3_t	cross_product(vec3_t &v1, vec3_t &v2);
ray_t	get_primary_ray(int x, int y, int sample);
vec3_t	get_sample_pos(int x, int y, int sample);
inline vec3_t	jitter(int x, int y, int s);
int 	ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp);
void 	load_scene(FILE *fp);

void yuv(int iter, int w, int h,
		uint8_t *r, uint8_t *g, uint8_t *b,
		uint8_t **y, uint8_t **u, uint8_t **v);

void rotate(int w, int h, uint8_t *yuv, Image **dstImg);

void output(int iter, Image *dstImg);

#endif /* BENCHAPPS_RAY_YUV_ROTATE_SEQ_COMMON_H_ */
