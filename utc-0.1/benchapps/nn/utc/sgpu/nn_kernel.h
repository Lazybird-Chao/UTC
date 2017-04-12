/*
 * kmeans_kernel.h
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_
#define BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_


template<typename T>
__global__ void distance_kernel(
		T *objects,
		int numCoords,
		int numObjs,
		T *targetObj,
		T *distanceObjs,
		int batchPerThread);



#endif /* BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_ */
