/*
 * c-ray_kernel.cc
 *
 * In this cuda kernel, we use one cuda thread to generate one pixel,
 * based on the scene parameters and apply the ray-tracing allgorith to
 * compute the outcome of one pixel
 *
 */

#include "c-ray_kernel.h"
#include "c-ray_kernel_device.h"

__device__ FTYPE aspect = 1.333333;
__device__ int xres_d;
__device__ int yres_d;

__global__ void render_kernel(
		int xres,
		int yres,
		unsigned int *pixels,
		int rays_per_pixel,
		sphere_array_t obj_array,
		int obj_count,
		vec3_t *lights,
		int lnum,
		vec2_t *urand,
		int *irand,
		camera_t cam){

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	xres_d = xres;
	yres_d = yres;
	int row = by*blockDim.y + ty;
	int colum = bx*blockDim.x + tx;
	if(row < yres && colum < xres){
		FTYPE rcp_samples = 1.0/(FTYPE)rays_per_pixel;
		double r, g, b;
		r = g = b = 0.0;
		ray_t tmpray;
		for(int s = 0; s<rays_per_pixel; s++){
			tmpray = get_primary_ray(colum, row, s, cam, urand, irand);
			vec3_t col = trace(tmpray,
					0,
					obj_array,
					obj_count,
					lights,
					lnum);
			r += col.x;
			g += col.y;
			b += col.z;
		}
		r = r * rcp_samples;
		g = g * rcp_samples;
		b = b * rcp_samples;

		pixels[row*xres + colum] = ((unsigned int)(MIN(r, 1.0) *255.0) & 0xff) << RSHIFT |
									((unsigned int)(MIN(g, 1.0) *255.0) & 0xff) << GSHIFT |
									((unsigned int)(MIN(b, 1.0) *255.0) & 0xff) << BSHIFT;
	}
	__syncthreads();
}

__device__ vec3_t trace(ray_t &ray,
		int depth,
		sphere_array_t &obj_array,
		int &obj_count,
		vec3_t *lights,
		int &lnum){
	vec3_t col;
	spoint_t sp, nearest_sp;
	int nearest_obj_idx = -1;
	int obj_iter = 0;

	if(depth >= MAX_RAY_DEPTH){
		col.x=col.y = col.z = 0.0;
		return col;
	}

	while(obj_iter<obj_count){
		if(ray_sphere(obj_array.pos[obj_iter],
				obj_array.rad[obj_iter],
				ray,
				&sp)){
			if(nearest_obj_idx==-1 || sp.dist < nearest_sp.dist){
				nearest_obj_idx = obj_iter;
				nearest_sp = sp;
			}
		}
		obj_iter++;
	}

	if(nearest_obj_idx != -1){
		col = shade(obj_array.mat[nearest_obj_idx],
				&nearest_sp,
				depth,
				obj_array,
				obj_count,
				lights,
				lnum);
	}
	else{
		col.x = col.y = col.z = 0.0;
	}
	return col;
}

__device__ vec3_t shade(material_t &obj_mat,
		spoint_t *sp,
		int depth,
		sphere_array_t &obj_array,
		int &obj_count,
		vec3_t *lights,
		int &lnum){
	vec3_t col = {0, 0, 0};
	for(int i=0; i<lnum; i++){
		FTYPE ispec, idiff;
		vec3_t ldir;
		ray_t shadow_ray;
		int obj_iter = 0;
		int in_shadow = 0;

		ldir.x = lights[i].x - sp->pos.x;
		ldir.y = lights[i].y - sp->pos.y;
		ldir.z = lights[i].z - sp->pos.z;

		shadow_ray.orig = sp->pos;
		shadow_ray.dir = ldir;

		/* shoot shadow rays to determine if we have a line of sight with the light */
		while(obj_iter < obj_count) {
			if(ray_sphere(obj_array.pos[obj_iter],
					obj_array.rad[obj_iter],
					shadow_ray,
					NULL)) {
				in_shadow = 1;
				break;
			}
			obj_iter++;
		}

		/* and if we're not in shadow, calculate direct illumination with the phong model. */
		if(!in_shadow) {
			NORMALIZE(ldir);

			idiff = MAX(DOT(sp->normal, ldir), 0.0);
			ispec = obj_mat.spow > 0.0 ? pow(MAX(DOT(sp->vref, ldir), 0.0), obj_mat.spow) : 0.0;

			col.x += idiff * obj_mat.col.x + ispec;
			col.y += idiff * obj_mat.col.y + ispec;
			col.z += idiff * obj_mat.col.z + ispec;
		}
	}

	if(obj_mat.refl > 0.0) {
		ray_t ray;
		vec3_t rcol;

		ray.orig = sp->pos;
		ray.dir = sp->vref;
		ray.dir.x *= RAY_MAG;
		ray.dir.y *= RAY_MAG;
		ray.dir.z *= RAY_MAG;

		rcol = trace(ray,
				depth + 1,
				obj_array,
				obj_count,
				lights,
				lnum);
		col.x += rcol.x * obj_mat.refl;
		col.y += rcol.y * obj_mat.refl;
		col.z += rcol.z * obj_mat.refl;
	}

	return col;

}

__device__ int ray_sphere(
		const vec3_t &obj_pos,
		FTYPE &obj_rad,
		ray_t &ray,
		spoint_t *sp){
	FTYPE a, b, c, d, sqrt_d, t1, t2;

	a = SQ(ray.dir.x) + SQ(ray.dir.y) + SQ(ray.dir.z);
	b = 2.0 * ray.dir.x * (ray.orig.x - obj_pos.x) +
				2.0 * ray.dir.y * (ray.orig.y - obj_pos.y) +
				2.0 * ray.dir.z * (ray.orig.z - obj_pos.z);
	c = SQ(obj_pos.x) + SQ(obj_pos.y) + SQ(obj_pos.z) +
				SQ(ray.orig.x) + SQ(ray.orig.y) + SQ(ray.orig.z) +
				2.0 * (-obj_pos.x * ray.orig.x - obj_pos.y * ray.orig.y - obj_pos.z * ray.orig.z) - SQ(obj_rad);

	if((d = SQ(b) - 4.0 * a * c) < 0.0)
		return 0;

	sqrt_d = sqrt(d);
	t1 = (-b + sqrt_d) / (2.0 * a);
	t2 = (-b - sqrt_d) / (2.0 * a);

	if((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0))
		return 0;

	if(sp) {
		if(t1 < ERR_MARGIN) t1 = t2;
		if(t2 < ERR_MARGIN) t2 = t1;
		sp->dist = t1 < t2 ? t1 : t2;

		sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
		sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
		sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;

		sp->normal.x = (sp->pos.x - obj_pos.x) / obj_rad;
		sp->normal.y = (sp->pos.y - obj_pos.y) / obj_rad;
		sp->normal.z = (sp->pos.z - obj_pos.z) / obj_rad;

		sp->vref = reflect(ray.dir, sp->normal);
		NORMALIZE(sp->vref);
	}
	return 1;

}

__device__ ray_t get_primary_ray(
		int x,
		int y,
		int sample,
		camera_t &cam,
		vec2_t *urand,
		int *irand){
	ray_t ray;
	FTYPE m[3][3];
	vec3_t i, j = {0, 1, 0}, k, dir, orig, foo;

	k.x = cam.targ.x - cam.pos.x;
	k.y = cam.targ.y - cam.pos.y;
	k.z = cam.targ.z - cam.pos.z;
	NORMALIZE(k);

	i = cross_product(j, k);
	j = cross_product(k, i);
	m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
	m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
	m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;

	ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
	ray.dir = get_sample_pos(x, y, sample, urand, irand);
	ray.dir.z = 1.0 / HALF_FOV;
	ray.dir.x *= RAY_MAG;
	ray.dir.y *= RAY_MAG;
	ray.dir.z *= RAY_MAG;

	dir.x = ray.dir.x + ray.orig.x;
	dir.y = ray.dir.y + ray.orig.y;
	dir.z = ray.dir.z + ray.orig.z;
	foo.x = dir.x * m[0][0] + dir.y * m[0][1] + dir.z * m[0][2];
	foo.y = dir.x * m[1][0] + dir.y * m[1][1] + dir.z * m[1][2];
	foo.z = dir.x * m[2][0] + dir.y * m[2][1] + dir.z * m[2][2];

	orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + cam.pos.x;
	orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + cam.pos.y;
	orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + cam.pos.z;

	ray.orig = orig;
	ray.dir.x = foo.x + orig.x;
	ray.dir.y = foo.y + orig.y;
	ray.dir.z = foo.z + orig.z;

	return ray;
}

__device__ vec3_t get_sample_pos(
		int x,
		int y,
		int sample,
		vec2_t *urand,
		int *irand) {
	vec3_t pt;
	FTYPE sf = 0.0;

	if(sf == 0.0) {
		sf = 1.5 / (FTYPE)xres_d;
	}

	pt.x = ((FTYPE)x / (FTYPE)xres_d) - 0.5;
	pt.y = -(((FTYPE)y / (FTYPE)yres_d) - 0.65) / aspect;

	if(sample) {
		vec3_t jt = jitter(x, y, sample, urand, irand);
		pt.x += jt.x * sf;
		pt.y += jt.y * sf / aspect;
	}
	return pt;
}












