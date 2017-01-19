/*
 * kmeans_kernel.h
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_
#define BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_

#define FTYPE float


__global__ void kmeans_kernel(
		T *object,
		int numCoords,
		int numObjs,
		int numClusters,
		T *clusters,
		T *membership,
		int batchPerThread);



#endif /* BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_ */
