/*
 * kmeans_kernel_v2.h
 *
 *  Created on: Jan 19, 2017
 *      Author: chao
 */

#ifndef KMEANS_KERNEL_V2_H_
#define KMEANS_KERNEL_V2_H_

#define FTYPE float

__global__ void membership_kernel(
		FTYPE *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		FTYPE *clusters,
		int *membership,
		int batchPerThread,
		FTYPE *new_clusters_reduce,
		int *new_clusters_size_reduce,
		int *change_count_reduce);


__global__ void new_clusters_size_kernel(
		int	*new_clusters_size_reduce,
		int	*new_clusters_size,
		int numClusters,
		int reduceSize);

__global__ void new_clusters_kernel(
		FTYPE *new_clusters_reduce,
		FTYPE *new_clusters,
		int *new_clusters_size,
		int numClusters,
		int numCoords,
		int reduceSize);


#endif /* KMEANS_KERNEL_V2_H_ */
