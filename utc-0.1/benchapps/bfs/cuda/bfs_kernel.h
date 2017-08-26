/*
 * bfs_kernel.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include "bfs_comm_data.h"

#define MAX_THREAD_PER_BLOCK 256   //change this smaller when graph is large
#define MAX_WAVE_SIZE	(1024*1024)//(total_graph_nodes/100)

//__device__ extern int nextWaveSize_d;


__global__ void bfs_singleblock(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave,
		int* nextWaveSize_d);

__global__ void bfs_multiblocks(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave,
		int* nextWaveSize_d);




#endif /* BFS_KERNEL_H_ */
