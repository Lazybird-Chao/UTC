/*
 * utc_kmeans_gpu_kernel.cc
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 *
 *
 *
 */

#include "utc_kmeans_gpu_kernel.h"


/*
 * compute the new membership of each object
 * one cuda thread compute for batch objs
 * both block and grid are 1-d
 */
__global__ void computeNewMembership_kernel(
		float* objs,
		float *clusters,
		int* membership,
		int numObjs,
		int numClusters,
		int numCoords,
		int batch){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	for(int i=0; i<batch; i++){
		int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
		if(obj_pos<numObjs){
			int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters);
			membership[obj_pos] = idx;
		}
	}

	/*int obj_s = (bx*blockDim.x + tx)*batch;
	int obj_e = obj_s + batch;
	for(int i= obj_s; i< obj_e; i++){
		if(i<numObjs){
			int idx = find_nearest_cluster(numClusters, numCoords, &objs[i*numCoords], clusters);
			membership[i] = idx;
		}
	}*/
	__syncthreads();
}




