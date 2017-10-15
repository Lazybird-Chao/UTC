/*
 * kmeans_kernel.h
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_
#define BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_


template<typename T>
__global__ void kmeans_kernel(
		T *objects,
		int numCoords,
		int numObjs,
		int numClusters,
		T *clusters,
		int *membership,
		int batchPerThread,
		int startObjIndex);



#endif /* BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_ */
