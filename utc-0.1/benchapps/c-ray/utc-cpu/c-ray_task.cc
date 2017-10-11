/*
 * c-ray_task.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#include "task.h"
#include <cmath>

thread_local int crayWorker::local_yres;
thread_local int crayWorker::local_startYresIndex;

void crayWorker::initImpl(global_vars g_vars,
		sphere2_t* obj_array,
		uint32_t *pixels,
		vec3_t *lights){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";

		this->g_vars = g_vars;
		this->obj_array = obj_array;
		this->pixels = pixels;
		this->lights = lights;
	}
	__fastIntraSync.wait();
	int yresPerThread = g_vars.yres / __numLocalThreads;
	if(__localThreadId < g_vars.yres % __numLocalThreads){
		local_yres = yresPerThread +1;
		local_startYresIndex = __localThreadId *(yresPerThread+1);
	}
	else{
		local_yres = yresPerThread;
		local_startYresIndex = __localThreadId*yresPerThread + g_vars.yres % __numLocalThreads;
	}
	__fastIntraSync.wait();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

void crayWorker::runImpl(double runtime[][1]){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;
	int xres = g_vars.xres;	//column
	int yres = g_vars.yres;	//row

	//vec2_t urand[NRAN];
	//int irand[NRAN];
	for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

	__fastIntraSync.wait();
	timer.start();
	for(int i = local_startYresIndex; i < local_startYresIndex+local_yres; i++){
		render_scanline(xres,
						yres,
						i,
						(uint32_t*)(pixels+i*xres),
						g_vars.rays_per_pixel);
	}
	double totaltime = timer.stop();

	__fastIntraSync.wait();
	runtime[__localThreadId][0] = totaltime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

void crayWorker::render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples){
	int i, s;
	FTYPE rcp_samples = 1.0 / (FTYPE)samples;

	for(i=0; i<xsz; i++) {
		double r, g, b;
		r = g = b = 0.0;

		for(s=0; s<samples; s++) {
			vec3_t col = trace(get_primary_ray(i, sl, s), 0);
			r += col.x;
			g += col.y;
			b += col.z;
		}

		r = r * rcp_samples;
		g = g * rcp_samples;
		b = b * rcp_samples;

		fb[i] = ((uint32_t)(MIN(r, 1.0) * 255.0) & 0xff) << RSHIFT |
							((uint32_t)(MIN(g, 1.0) * 255.0) & 0xff) << GSHIFT |
							((uint32_t)(MIN(b, 1.0) * 255.0) & 0xff) << BSHIFT;
	}
}

vec3_t crayWorker::trace(ray_t ray, int depth) {
	vec3_t col;
	spoint_t sp, nearest_sp;
	sphere2_t *nearest_obj=nullptr;
	int obj_iter = 0;

	if(depth >= MAX_RAY_DEPTH){
		col.x = col.y = col.z = 0.0;
		return col;
	}

	/* find the nearest intersection ... */
	while(obj_iter < g_vars.obj_count) {
		if(ray_sphere(&obj_array[obj_iter], ray, &sp)) {
			if(!nearest_obj || sp.dist < nearest_sp.dist) {
				nearest_obj = &obj_array[obj_iter];
				nearest_sp = sp;
			}
		}
		obj_iter++;
	}

	/* and perform shading calculations as needed by calling shade() */
	if(nearest_obj) {
		col = shade(nearest_obj, &nearest_sp, depth);
	} else {
		col.x = col.y = col.z = 0.0;
	}

	return col;
}

int crayWorker::ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp) {
	FTYPE a, b, c, d, sqrt_d, t1, t2;

	a = SQ(ray.dir.x) + SQ(ray.dir.y) + SQ(ray.dir.z);
	b = 2.0 * ray.dir.x * (ray.orig.x - sph->pos.x) +
				2.0 * ray.dir.y * (ray.orig.y - sph->pos.y) +
				2.0 * ray.dir.z * (ray.orig.z - sph->pos.z);
	c = SQ(sph->pos.x) + SQ(sph->pos.y) + SQ(sph->pos.z) +
				SQ(ray.orig.x) + SQ(ray.orig.y) + SQ(ray.orig.z) +
				2.0 * (-sph->pos.x * ray.orig.x - sph->pos.y * ray.orig.y - sph->pos.z * ray.orig.z) - SQ(sph->rad);

	if((d = SQ(b) - 4.0 * a * c) < 0.0) return 0;

	sqrt_d = sqrt(d);
	t1 = (-b + sqrt_d) / (2.0 * a);
	t2 = (-b - sqrt_d) / (2.0 * a);

	if((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0)) return 0;

	if(sp) {
		if(t1 < ERR_MARGIN) t1 = t2;
		if(t2 < ERR_MARGIN) t2 = t1;
		sp->dist = t1 < t2 ? t1 : t2;

		sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
		sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
		sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;

		sp->normal.x = (sp->pos.x - sph->pos.x) / sph->rad;
		sp->normal.y = (sp->pos.y - sph->pos.y) / sph->rad;
		sp->normal.z = (sp->pos.z - sph->pos.z) / sph->rad;

		sp->vref = reflect(ray.dir, sp->normal);
		NORMALIZE(sp->vref);
	}
	return 1;

}

vec3_t crayWorker::shade( sphere2_t *obj, spoint_t *sp, int depth) {
	int i;
	vec3_t col = {0,0,0};

	/* for all lights ... */
	for(i=0; i<g_vars.lnum; i++) {
		FTYPE ispec, idiff;
		vec3_t ldir;
		ray_t shadow_ray;
		int obj_iter = 0;//struct sphere *iter = obj_list->next;
		int in_shadow = 0;

		ldir.x = lights[i].x - sp->pos.x;
		ldir.y = lights[i].y - sp->pos.y;
		ldir.z = lights[i].z - sp->pos.z;

		shadow_ray.orig = sp->pos;
		shadow_ray.dir = ldir;

		/* shoot shadow rays to determine if we have a line of sight with the light */
		while(obj_iter < g_vars.obj_count) {
			if(ray_sphere(&obj_array[obj_iter], shadow_ray, 0)) {
				in_shadow = 1;
				break;
			}
			obj_iter++;
		}

		/* and if we're not in shadow, calculate direct illumination with the phong model. */
		if(!in_shadow) {
			NORMALIZE(ldir);

			idiff = MAX(DOT(sp->normal, ldir), 0.0);
			ispec = obj->mat.spow > 0.0 ? pow(MAX(DOT(sp->vref, ldir), 0.0), obj->mat.spow) : 0.0;

			col.x += idiff * obj->mat.col.x + ispec;
			col.y += idiff * obj->mat.col.y + ispec;
			col.z += idiff * obj->mat.col.z + ispec;
		}
	}

	/* Also, if the object is reflective, spawn a reflection ray, and call trace()
	 * to calculate the light arriving from the mirror direction.
	 */
	if(obj->mat.refl > 0.0) {
		ray_t ray;
		vec3_t rcol;

		ray.orig = sp->pos;
		ray.dir = sp->vref;
		ray.dir.x *= RAY_MAG;
		ray.dir.y *= RAY_MAG;
		ray.dir.z *= RAY_MAG;

		rcol = trace(ray, depth + 1);
		col.x += rcol.x * obj->mat.refl;
		col.y += rcol.y * obj->mat.refl;
		col.z += rcol.z * obj->mat.refl;
	}

	return col;
}

/* determine the primary ray corresponding to the specified pixel (x, y) */
ray_t crayWorker::get_primary_ray(int x, int y, int sample) {
	ray_t ray;
	FTYPE m[3][3];
	vec3_t i, j = {0, 1, 0}, k, dir, orig, foo;

	k.x = g_vars.cam.targ.x - g_vars.cam.pos.x;
	k.y = g_vars.cam.targ.y - g_vars.cam.pos.y;
	k.z = g_vars.cam.targ.z - g_vars.cam.pos.z;
	NORMALIZE(k);

	i = cross_product(j, k);
	j = cross_product(k, i);
	m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
	m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
	m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;

	ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
	ray.dir = get_sample_pos(x, y, sample);
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

	orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + g_vars.cam.pos.x;
	orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + g_vars.cam.pos.y;
	orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + g_vars.cam.pos.z;

	ray.orig = orig;
	ray.dir.x = foo.x + orig.x;
	ray.dir.y = foo.y + orig.y;
	ray.dir.z = foo.z + orig.z;

	return ray;
}


vec3_t crayWorker::get_sample_pos(int x, int y, int sample) {
	vec3_t pt;
	static FTYPE sf = 0.0;

	if(sf == 0.0) {
		sf = 1.5 / (FTYPE)g_vars.xres;
	}

	pt.x = ((FTYPE)x / (FTYPE)g_vars.xres) - 0.5;
	pt.y = -(((FTYPE)y / (FTYPE)g_vars.yres) - 0.65) / g_vars.aspect;

	if(sample) {
		vec3_t jt = jitter(x, y, sample);
		pt.x += jt.x * sf;
		pt.y += jt.y * sf / g_vars.aspect;
	}
	return pt;
}

/* jitter function taken from Graphics Gems I. */
inline vec3_t crayWorker::jitter(int x, int y, int s) {
	vec3_t pt;
	pt.x = urand[(x + (y << 2) + irand[(x + s) & MASK]) & MASK].x;
	pt.y = urand[(y + (x << 2) + irand[(y + s) & MASK]) & MASK].y;
	return pt;
}


/*
 * some help routine
 */
inline vec3_t crayWorker::reflect(vec3_t &v, vec3_t &n) {
	vec3_t res;
	FTYPE dot = v.x * n.x + v.y * n.y + v.z * n.z;
	res.x = -(2.0 * dot * n.x - v.x);
	res.y = -(2.0 * dot * n.y - v.y);
	res.z = -(2.0 * dot * n.z - v.z);
	return res;
}

inline vec3_t crayWorker::cross_product(vec3_t &v1, vec3_t &v2) {
	vec3_t res;
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
	return res;
}



