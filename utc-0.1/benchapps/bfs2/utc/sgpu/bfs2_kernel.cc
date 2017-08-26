/*
 * bfs_kernel.cc
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 *
 * The singleblock kernel is used when the number of nodes in frontwave is small,
 * can only feed one cudablock. Ther other kernel is for large frontwave case.
 * Because there's no global threads sync in CUDA, that's why we differ these two
 * cases.
 *
 */

#include "bfs2_kernel.h"

#include <limits.h>
#include <stdio.h>


__global__ void bfs_findFront(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		uint8_t* frontWave,
		uint8_t* nextWave,
		int total_nodes,
		int batch){

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tid = bx*blockDim.x + tx;

	int startpos = bx*blockDim.x*batch;
	for(int i=0; i<batch; i++){
		int index = startpos + tx + i*blockDim.x;
		if(index<total_nodes){
			if(frontWave[index] == 1){
				frontWave[index] = 0;
				int level = spath[index];
				Node_t cur_node = nodes[index];
				for(int j= cur_node.start_edgeid;
						j<cur_node.start_edgeid + cur_node.num_edges;
						j++){
					int neighbour = edges[j];
					if(spath[neighbour] == INT_MAX){
						nextWave[neighbour]= 1;
						spath[neighbour] = level+1;
					}
				}
			}
		}
	}

}

__global__ void bfs_updateFront(
		uint8_t* stopflag,
		uint8_t* frontWave,
		uint8_t* nextWave,
		int total_nodes,
		int batch){

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tid = bx*blockDim.x + tx;

	int startpos = bx*blockDim.x*batch;
	for(int i=0; i<batch; i++){
		int index = startpos + tx + i*blockDim.x;
		if(index<total_nodes){
			if(nextWave[index]==1){
				frontWave[index] = 1;
				nextWave[index] = 0;
				if(*stopflag !=0)
					*stopflag = 0;
			}
		}
	}

}




