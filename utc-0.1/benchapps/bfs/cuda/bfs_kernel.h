/*
 * bfs_kernel.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include "bfs_comm_data.h"


extern __device__ int nextWaveSize_d;


__global__ void bfs_singleblock(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave);

__global__ void bfs_multiblocks(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave);



#endif /* BFS_KERNEL_H_ */
