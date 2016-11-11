/*
 * utc_kmeans_gpu_kernel.h
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 */

#ifndef UTC_KMEANS_GPU_KERNEL_H_
#define UTC_KMEANS_GPU_KERNEL_H_

__global__ void computeNewMembership_kernel(
		float* objs,
		float* clusters,
		int* membership,
		int numObjs,
		int numClusters,
		int numCoords,
		int batch);


__device__ __host__  inline float euclid_dist_2(
		int    numdims,  /* no. dimensions */
        float *coord1,   /* [numdims] */
        float *coord2)   /* [numdims] */
{
	int i;
	float ans=0.0;

	for (i=0; i<numdims; i++)
	ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
	//ans = sqrt(ans);
	return(ans);
}

__device__ __host__  inline int find_nearest_cluster(
		int     numClusters, /* no. clusters */
        int     numCoords,   /* no. coordinates */
        float  *object,      /* [numCoords] */
        float  *clusters)    /* [numClusters][numCoords] */
{
	int   index, i;
	float dist, min_dist;

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


#endif /* UTC_KMEANS_GPU_KERNEL_H_ */
