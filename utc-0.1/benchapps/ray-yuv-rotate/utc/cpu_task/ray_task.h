/*
 * ray_task.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef RAY_TASK_H_
#define RAY_TASK_H_

#include "Utc.h"
#include "../common.h"
#include <cmath>

class crayCPUWorker: public UserTaskBase{
private:
	global_vars g_vars;
	sphere2_t* obj_array;
	vec3_t *lights;
	uint32_t *pixels_array;

	static thread_local int local_yres;
	static thread_local int local_startYresIndex;

	vec2_t urand[NRAN];
	int irand[NRAN];

	void render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples);
	vec3_t trace(ray_t ray, int depth);
	int ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp);
	vec3_t shade( sphere2_t *obj, spoint_t *sp, int depth);
	ray_t get_primary_ray(int x, int y, int sample);
	vec3_t get_sample_pos(int x, int y, int sample);
	vec3_t jitter(int x, int y, int s);
	vec3_t reflect(vec3_t &v, vec3_t &n);
	vec3_t cross_product(vec3_t &v1, vec3_t &v2);

	iUtc::Conduit *m_cdtOut;

public:
	void initImpl(global_vars g_vars,
			sphere2_t* obj_array,
			uint32_t *pixels,
			vec3_t *lights,
			iUtc::Conduit *cdtOut);

	void runImpl(double runtime[][3], int loop, bool needDoOutput);

};




#endif /* RAY_TASK_H_ */
