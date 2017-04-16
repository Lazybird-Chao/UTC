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

template <typename T>
__global__ void topk_kernel(
		int numObjs,
		int numNN,
		T *distanceObjs,
		int *topkIndexArray);

#endif /* BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_ */
