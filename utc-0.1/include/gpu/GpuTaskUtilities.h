/*
 * GpuTaskUtilities.h
 *
 *  Created on: Oct 31, 2016
 *      Author: chao
 */

#ifndef UTC_GPUTASKUTILITIES_H_
#define UTC_GPUTASKUTILITIES_H_

#include "UtcGpuBasics.h"
#include "UtcGpuContext.h"

namespace iUtc{

	UtcGpuContext* getCurrentUtcGpuCtx();

	int getCurrentUtcGpuId();

	int getCurrentCudaDeviceId();

	cudaStream_t getCurrentStream();

	//CUcontext* getCurrentCudaCtx();

}



#endif /* UTC_GPUTASKUTILITIES_H_ */
