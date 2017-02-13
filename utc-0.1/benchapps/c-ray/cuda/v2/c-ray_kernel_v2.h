/*
 * c-ray_kernel_v2.h
 *
 *  Created on: Feb 13, 2017
 *      Author: chao
 */

#ifndef C_RAY_KERNEL_V2_H_
#define C_RAY_KERNEL_V2_H_

/*
 * required data structure
 */
#define FTYPE double
#ifdef FTYPE
#define SFLOAT 1
#define DFLOAT 2
#endif

#if SFLOAT==1
	typedef double3	 vec3_t;
	typedef double2	 vec2_t;
#else
	typedef float3   vec2_t;
	typedef float2	 vec2_t;
#endif

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
	vec3_t	pos, targ;
	FTYPE	fov;
}	camera_t;

struct global_vars{
	int xres;
	int yres;
	int lnum;
	int obj_count;
	int rays_per_pixel;
	FTYPE aspect;
	camera_t cam;
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
#define MAX(a, b)	((a) > (b) ? (a) : (b))
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


/*
 * global vars for device functions
 */
extern __device__ __const__ struct global_vars g_vars_d;
extern __device__ __const__ vec3_t lights_d[MAX_LIGHTS];
extern __device__ __const__ vec2_t urand_d[NRAN];
extern __device__ __const__ int irand_d[NRAN];


/*
 * cuda kernels
 */
__global__ void render_kernel(
		unsigned int *pixels,
		sphere_array_t obj_array);



#endif /* C_RAY_KERNEL_V2_H_ */
