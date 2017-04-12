/*
 * kmeans_kernel.cc
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#include "nn_kernel.h"

template<typename T>
__global__ void distance_kernel(
		T *objs,
		int numCoords,
		int numObjs,
		T *targetObj,
		T *distanceObjs,
		int batchPerThread){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	for(int i=0; i<batchPerThread; i++){
		int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
		if(obj_pos<numObjs){
			T *coord1 = &objs[obj_pos*numCoords];
			T ans=0.0;
			for (i=0; i<numCoords; i++)
				ans += (coord1[i]-targetObj[i]) * (coord1[i]-targetObj[i]);
			distanceObjs[obj_pos] = sqrt(ans);
		}
	}

}


template
__global__ void distance_kernel<float>(
		float *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		float *clusters,
		int *membership,
		int batchPerThread);
template
__global__ void distance_kernel<double>(
		double *objs,
		int numCoords,
		int numObjs,
		int numClusters,
		double *clusters,
		int *membership,
		int batchPerThread);
