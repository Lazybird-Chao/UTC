/*
 * bfs_kernel.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_KERNEL_H_
#define BFS_KERNEL_H_

#include "../bfs2_comm_data.h"
#include "stdint.h"


__global__ void bfs_findFront(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		uint8_t* frontWave,
		uint8_t* nextWave,
		int total_nodes,
		int batch);

__global__ void bfs_updateFront(
		uint8_t* stopflag,
		uint8_t* frontWave,
		uint8_t* nextWave,
		int total_nodes,
		int batch);




#endif /* BFS_KERNEL_H_ */
