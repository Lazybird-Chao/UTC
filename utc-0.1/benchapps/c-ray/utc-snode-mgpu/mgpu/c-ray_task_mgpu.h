/*
 * c-ray_task_sgpu.h
 *
 *  Created on: Mar 24, 2017
 *      Author: chao
 */

#ifndef C_RAY_TASK_SGPU_H_
#define C_RAY_TASK_SGPU_H_

#include "../typeconfig.h"

#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

class crayMGPU: public UserTaskBase{
private:
	global_vars g_vars;
	sphere_array_t obj_array;
	vec3_t *lights;
	uint32_t *pixels;

	static thread_local int local_yres;
	static thread_local int local_startYresIndex;

public:
	void initImpl(global_vars g_vars,
			sphere_array_t obj_array,
			uint32_t *pixels,
			vec3_t *lights);

	void runImpl(double runtime[][4], MemType memtype);

};



#endif /* C_RAY_TASK_SGPU_H_ */
