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
		FTYPE *objects,
		int numCoords,
		int numObjs,
		int numClusters,
		FTYPE *clusters,
		int *membership,
		int batchPerThread);



#endif /* BENCHAPPS_KMEANS_CUDA_KMEANS_KERNEL_H_ */
