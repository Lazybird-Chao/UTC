/*
 * kmeans_kernel_v2.cc
 *
 *  Created on: Jan 19, 2017
 *      Author: chao
 */

#include "stdio.h"
#include "kmeans_kernel_v2.h"

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

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

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
		int *change_count_reduce){
	int bx = blockIdx.x;
	int tx = threadIdx.x;


	__shared__ unsigned int change_count[256];
	change_count[tx] =0;
	unsigned int changeMax = numObjs +1;

	if(numClusters<= blockDim.x && numClusters*numCoords < 1024){
		__shared__ FTYPE clusters_s[1024];
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
					if(membership[obj_pos] != idx){
						membership[obj_pos] = idx;
						//atomicInc(&change_count, changeMax);
						change_count[tx]++;
					}

					atomicAdd((unsigned int *)&new_clusters_size_reduce[idx+bx*numClusters], 1);
					for(int j=0; j<numCoords; i++){
						atomicAdd(&new_clusters_reduce[bx*numClusters*numCoords + idx*numCoords + j], objs[obj_pos*numCoords + j]);
					}
				}
			}
	}
	else{
		for(int i=0; i<batchPerThread; i++){
				int obj_pos = bx*blockDim.x + tx + i*blockDim.x*gridDim.x;
				if(obj_pos<numObjs){
					int idx = find_nearest_cluster(numClusters, numCoords, &objs[obj_pos*numCoords], clusters);
					if(membership[obj_pos] != idx){
						membership[obj_pos] = idx;
						//atomicInc(&change_count, changeMax);
						change_count[tx]++;
					}

					atomicAdd((unsigned int *)&new_clusters_size_reduce[idx+bx*numClusters], 1);
					for(int j=0; j<numCoords; i++){
						atomicAdd(&new_clusters_reduce[bx*numClusters*numCoords + idx*numCoords + j], objs[obj_pos*numCoords + j]);
					}
				}
		}
	}
	__syncthreads();
	//printf("%d %d\n", tx, bx);

	if(tx ==0){
		int sum=0;
		for(int i=0; i<blockDim.x; i++)
			sum += change_count[i];
		change_count_reduce[bx] = sum;
	}

}


__global__ void new_clusters_size_kernel(
		int	*new_clusters_size_reduce,
		int	*new_clusters_size,
		int numClusters,
		int reduceSize){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int idx = tx + bx*blockDim.x;
	int sum = 0;
	if( idx < numClusters){
		for(int i=0; i<reduceSize; i++)
			sum +=new_clusters_size_reduce[i*numClusters + idx];
		new_clusters_size[idx] = sum;
	}
	__syncthreads();
}


__global__ void new_clusters_kernel(
		FTYPE *new_clusters_reduce,
		FTYPE *new_clusters,
		int *new_clusters_size,
		int numClusters,
		int numCoords,
		int reduceSize){
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int idx = tx + bx*blockDim.x;
	FTYPE sum = 0;
	if(idx < numClusters*numCoords){
		int csize = new_clusters_size[idx/numCoords];
		for(int i=0; i<reduceSize; i++)
			sum += new_clusters_reduce[i*numClusters*numCoords +idx];
		if(csize >1)
			sum /= csize;
		new_clusters[idx] = sum;
	}
	__syncthreads();

}



