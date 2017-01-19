/*
 * kmeans_kernel.cc
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#include "kmeans_kernel.h"

__device__  inline float euclid_dist_2(
		int    numdims,  /* no. dimensions */
		FTYPE *coord1,   /* [numdims] */
		FTYPE *coord2)   /* [numdims] */
{
	int i;
	FTYPE ans=0.0;

	for (i=0; i<numdims; i++)
		ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
	//ans = sqrt(ans);
	return(ans);
}

__device__  inline int find_nearest_cluster(
		int     numClusters, /* no. clusters */
        int     numCoords,   /* no. coordinates */
        FTYPE  *object,      /* [numCoords] */
		FTYPE  *clusters)    /* [numClusters][numCoords] */
{
	int   index, i;
	FTYPE dist, min_dist;

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


__global__ void kmeans_kernel(
		FTYPE *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		FTYPE *clusters,
		int *membership,
		int batchPerThread){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	if(numClusters<= blodkDim.x && numClusters*numCoords < 1024){
		__shared__ FTYPE clusters_s[1024];
		if(tx < numClusters){
			for(int i=0; i<numCoords; i++){
				clusters_s[tx*numCoords + i] = clusters[tx*numCoords + i];
			}
		}
		__syncthreads();

		for(int i=0; i<batch; i++){
			int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
			if(obj_pos<numObjs){
				int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters_s);
				membership[obj_pos] = idx;
			}
		}
	}
	else{
		for(int i=0; i<batch; i++){
			int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
			if(obj_pos<numObjs){
				int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters);
				membership[obj_pos] = idx;
			}
		}
	}
	__syncthreads();

}

