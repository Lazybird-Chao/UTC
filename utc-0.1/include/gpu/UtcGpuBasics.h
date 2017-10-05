/*
 * UtcGpuBasics.h
 *
 *  Created on: Oct 21, 2016
 *      Author: chao
 *
 *  Basic configuration infos
 *
 */

#ifndef UTC_GPU_UTCGPUBASICS_H_
#define UTC_GPU_UTCGPUBASICS_H_

#include "UtcBasics.h"

//#define ERROR_LINE " line:"<<__LINE__<<", file:"<<__FILE__<<" "

//#define ENABLE_GPU_TASK		1

#define ENABLE_CONCURRENT_CUDA_KERNEL	1

#if ENABLE_CONCURRENT_CUDA_KERNEL
	#define USING_NONBLOCKING_STREAM	0
#endif

/*
 *  1	cudaCtxMapToThread,
	2	cudaCtxMapToTask,
	3	cudaCtxMapToDevice
 */
#define CUDA_CONTEXT_MAP_MODE	3

#define MAX_DEVICE_PER_NODE		16

#define CUDA_MAJOR	65


#define CHECK_GPU_ABILITY	0

#define ENABLE_GLOBAL_GPU_DATA	1


#endif /* UTC_GPU_UTCGPUBASICS_H_ */
