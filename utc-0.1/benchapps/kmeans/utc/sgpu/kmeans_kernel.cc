/*
 * kmeans_kernel.cc
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#include "kmeans_kernel.h"

template<typename T>
__device__  inline T euclid_dist_2(
		int    numdims,  /* no. dimensions */
		T *coord1,   /* [numdims] */
		T *coord2)   /* [numdims] */
{
	int i;
	T ans=0.0;

	for (i=0; i<numdims; i++)
		ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
	//ans = sqrt(ans);
	return(ans);
}

template<typename T>
__device__  inline int find_nearest_cluster(
		int     numClusters, /* no. clusters */
        int     numCoords,   /* no. coordinates */
        T  *object,      /* [numCoords] */
		T  *clusters)    /* [numClusters][numCoords] */
{
	int   index, i;
	T dist, min_dist;

	/* find the cluster id that has min distance to object */
	index    = 0;
	min_dist = euclid_dist_2(numCoords, object, &clusters[0]);

	for (i=1; i<numClusters; i++) {
		dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
		/* no need square root */
		if (dist < min_dist) { /* find the min and its array index */
			min_dist = dist;
			index    = i;
		}
	}
	return(index);
}


template<typename T>
__global__ void kmeans_kernel(
		T *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		T *clusters,
		int *membership,
		int batchPerThread){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	if(numClusters<= blockDim.x && numClusters*numCoords <= 1024){
		__shared__ T clusters_s[1024];
		if(tx < numClusters){
			for(int i=0; i<numCoords; i++){
				clusters_s[tx*numCoords + i] = clusters[tx*numCoords + i];
			}
		}
		__syncthreads();

		for(int i=0; i<batchPerThread; i++){
			int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
			if(obj_pos<numObjs){
				int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters_s);
				membership[obj_pos] = idx;
			}
		}
	}
	else{
		for(int i=0; i<batchPerThread; i++){
			int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
			if(obj_pos<numObjs){
				int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters);
				membership[obj_pos] = idx;
			}
		}
	}
	__syncthreads();

}


template
__global__ void kmeans_kernel<float>(
		float *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		float *clusters,
		int *membership,
		int batchPerThread);
template
__global__ void kmeans_kernel<double>(
		double *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		double *clusters,
		int *membership,
		int batchPerThread);
