/*
 * c-ray_kernel_device.h
 *
 *      Author: Chao
 */

#ifndef BENCHAPPS_C_RAY_CUDA_C_RAY_KERNEL_DEVICE_H_
#define BENCHAPPS_C_RAY_CUDA_C_RAY_KERNEL_DEVICE_H_

/*
 * other device functions
 */
__device__ vec3_t trace(ray_t &ray,
		int depth,
		sphere_array_t &obj_array,
		int &obj_count,
		vec3_t *lights,
		int &lnum);

__device__ vec3_t shade(material_t &obj_mat,
		spoint_t *sp,
		int depth,
		sphere_array_t &obj_array,
		int &obj_count,
		vec3_t *lights,
		int &lnum);

__device__ int ray_sphere(
		const vec3_t &obj_pos,
		FTYPE &obj_rad,
		ray_t &ray,
		spoint_t *sp);

__device__ ray_t get_primary_ray(
		int x,
		int y,
		int sample,
		camera_t &cam,
		vec2_t *urand,
		int *irand);

__device__ vec3_t get_sample_pos(
		int x,
		int y,
		int sample,
		vec2_t *urand,
		int *irand);

/* jitter function taken from Graphics Gems I. */
__device__ inline vec3_t jitter(int x, int y, int s, vec2_t *urand, int *irand) {
	vec3_t pt;
	pt.x = urand[(x + (y << 2) + irand[(x + s) & MASK]) & MASK].x;
	pt.y = urand[(y + (x << 2) + irand[(y + s) & MASK]) & MASK].y;
	return pt;
};


/*
 * some help routine
 */
__device__ inline vec3_t reflect(vec3_t &v, vec3_t &n) {
	vec3_t res;
	FTYPE dot = v.x * n.x + v.y * n.y + v.z * n.z;
	res.x = -(2.0 * dot * n.x - v.x);
	res.y = -(2.0 * dot * n.y - v.y);
	res.z = -(2.0 * dot * n.z - v.z);
	return res;
};

__device__ inline vec3_t cross_product(vec3_t &v1, vec3_t &v2) {
	vec3_t res;
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
	return res;
};




#endif /* BENCHAPPS_C_RAY_CUDA_C_RAY_KERNEL_DEVICE_H_ */
