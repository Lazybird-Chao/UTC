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


#define ENABLE_GPU_TASK		1

#define ENABLE_CONCURRENT_CUDA_KERNEL	0

/*
 *  1	cudaCtxMapToThread,
	2	cudaCtxMapToTask,
	3	cudaCtxMapToDevice
 */
#define CUDA_CONTEX_MAP_MODE	3

#define MAX_DEVICE_PER_NODE		4

#define CUDA_MAJOR	55


#endif /* UTC_GPU_UTCGPUBASICS_H_ */