/*
 * common.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cmath>


#define FTYPE float


typedef struct vec3{
	FTYPE x,y,z;
public:
	vec3(){
	}
	__host__ __device__ vec3(FTYPE a, FTYPE b, FTYPE c){
		x=a;
		y=b;
		z=c;
	}
}	vec3_t;

typedef struct vec2{
	FTYPE x,y;
public:
	vec2(){
	}
}	vec2_t;

typedef struct ray{
	vec3_t orig, dir;
}	ray_t;

typedef struct material{
	vec3_t 	col;
	FTYPE		spow;
	FTYPE		refl;
}	material_t;

typedef struct sphere{
	vec3_t		pos;
	material_t	mat;
	FTYPE			rad;
	struct sphere *next;
}	sphere_t;

typedef struct sphere_array{
	vec3_t		*pos;
	material_t	*mat;
	FTYPE			*rad;
}	sphere_array_t; //9 units

typedef struct spoint{
	vec3_t	pos, normal, vref;
	FTYPE		dist;
}	spoint_t;

typedef struct camera{
	vec3_t	pos;
	vec3_t targ;
	FTYPE	fov;
public:
	camera(){
	}
}	camera_t;

struct global_vars{
	int xres;
	int yres;
	int lnum;
	int obj_count;
	int rays_per_pixel;
	FTYPE aspect;
	camera_t cam;
public:
	global_vars(){
	}
};


/*
 * constant variables
 */
#define MAX_LIGHTS		16				/* maximum number of lights */
#define RAY_MAG			1000.0			/* trace rays of this magnitude */
#define MAX_RAY_DEPTH	5				/* raytrace recursion limit */
#define FOV				0.78539816		/* field of view in rads (pi/4) */
#define HALF_FOV		(FOV * 0.5)
#define ERR_MARGIN		1e-6			/* an arbitrary error margin to avoid surface acne */

/*
 * bit-shift ammount for packing each color into a 32bit uint
 */
#ifdef LITTLE_ENDIAN
#define RSHIFT	16
#define BSHIFT	0
#else	/* big endian */
#define RSHIFT	0
#define BSHIFT	16
#endif	/* endianess */
#define GSHIFT	8	/* this is the same in both byte orders */

/* some helpful macros... */
#define SQ(x)		((x) * (x))
#ifndef MAX
#define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif
#define MIN(a, b)	((a) < (b) ? (a) : (b))
#define DOT(a, b)	((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define NORMALIZE(a)  do {\
	FTYPE len = sqrt(DOT(a, a));\
	(a).x /= len; (a).y /= len; (a).z /= len;\
} while(0);

/*
 * global variables
 */
#define NRAN	1024
#define MASK	(NRAN - 1)






#endif /* COMMON_H_ */
