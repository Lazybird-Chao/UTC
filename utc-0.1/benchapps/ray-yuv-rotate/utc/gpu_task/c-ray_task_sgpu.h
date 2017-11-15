/*
 * c-ray_task_sgpu.h
 *
 *  Created on: Mar 24, 2017
 *      Author: chao
 */

#ifndef C_RAY_TASK_SGPU_H_
#define C_RAY_TASK_SGPU_H_

#include "../common.h"

#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

class craySGPU: public UserTaskBase{
private:
	global_vars g_vars;
	sphere_array_t obj_array;
	vec3_t *lights;
	uint32_t *pixels_array;

	Conduit *m_cdtOut;

public:
	void initImpl(global_vars g_vars,
			sphere_array_t obj_array,
			uint32_t *pixels_array,
			vec3_t *lights,
			Conduit *cdtOut);

	void runImpl(double runtime[][3], MemType memtype, int loop);

};



#endif /* C_RAY_TASK_SGPU_H_ */
