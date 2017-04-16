/*
 * kmeans_kernel.cc
 *
 *  Created on: Jan 18, 2017
 *      Author: Chao
 */

#include "nn_kernel.h"



/*
 * kernel to compute distance of each object point
 */
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
		float *targetObj,
		float *distanceObjs,
		int batchPerThread);
template
__global__ void distance_kernel<double>(
		double *objs,
		int numCoords,
		int numObjs,
		double *targetObj,
		double *distanceObjs,
		int batchPerThread);


/*
 * kernel to find max k values in parallel
 */
template <typename T>
__global__ void topk_kernel(
		int numObjs,
		int numNN,
		T *distanceObjs,
		int *topkIndexArray){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int objperblock = (numObjs+gridDim.x-1)/gridDim.x;
	int objstart = objperblock*bx;
	int objend = objstart + objperblock;
	if(objend > numObjs)
		objend = numObjs;

	int objperthread = (objend - objstart + blockDim.x-1)/blockDim.x;
	int objstart_t = objstart + tx;
	int objend_t = objstart_t + blockDim.x*(objperthread-1);
	if(objend_t > objend)
		objend_t = objend;
	__shared__ int localTopkIndexArray[64][16];
	for(int j=0; j<numNN; j++){
		int min = objstart_t;
		for(int i=objstart_t; i<objend_t; i+=blockDim.x){
			if((distanceObjs[min]<0) ||
					(distanceObjs[i]>0 &&
					distanceObjs[min] > distanceObjs[i]))
				min = i;
		}
		localTopkIndexArray[j][tx] = min;
		distanceObjs[min] *= -1;
	}
	for(int i=objstart_t; i<objend_t; i+=blockDim.x){
		if(distanceObjs[i] <0)
			distanceObjs[i] *= -1;

	}
	__syncthreads();
	int distanceStart = numNN*bx;
	int distanceEnd = numNN*(bx+1);
	if(tx==0){
		for(int j=0; j<numNN; j++){
			int min = j;
			for(int i=j+1; i<blockDim.x*numNN;i++){
				int row = i/blockDim.x;
				int col = i%blockDim.x;
				if(distanceObjs[localTopkIndexArray[min/blockDim.x][min%blockDim.x]]
				                   >distanceObjs[localTopkIndexArray[row][col]])
					min = i;
			}
			if(min!=j){
				int tmp = localTopkIndexArray[min/blockDim.x][min%blockDim.x];
				localTopkIndexArray[min/blockDim.x][min%blockDim.x] =
						localTopkIndexArray[j/blockDim.x][j%blockDim.x];
				localTopkIndexArray[j/blockDim.x][j%blockDim.x] = tmp;
			}
			topkIndexArray[distanceStart+j] = localTopkIndexArray[j/blockDim.x][j%blockDim.x];
		}
	}
	__syncthreads();
}

template
__global__ void topk_kernel<float>(
		int numObjs,
		int numNN,
		float *distanceObjs,
		int *topkIndexArray);
template
__global__ void topk_kernel<double>(
		int numObjs,
		int numNN,
		double *distanceObjs,
		int *topkIndexArray);



