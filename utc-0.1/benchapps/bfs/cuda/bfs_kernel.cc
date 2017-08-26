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

#include <limits.h>
#include <stdio.h>

#include "bfs_kernel.h"
#include "bfs_comm_data.h"

//__device__ int nextWaveSize_d;

__global__ void bfs_singleblock(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave,
		int* nextWaveSize_d){

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tid = bx*blockDim.x + tx;

	volatile __shared__ int nextPos;
	__shared__ int currentWave_s[2048];
	__shared__ int nextWave_s[2048];
	int *pcurr = currentWave_s;
	int *pnext = nextWave_s;

	/*if(tx==0 && nextWaveSize_d !=0)
		printf("%d %d %d\n", bx, tid, nextWaveSize_d);*/

	if(tid<frontWaveSize)
		currentWave_s[tid] = frontWave[tid];

	int curWsize = frontWaveSize;
	do{
		if(tx==0){
			nextPos = 0;
		}
		__syncthreads();

		int offset = tid;
		while(offset<curWsize){
			Node_t curNode = nodes[pcurr[offset]];
			int level = spath[pcurr[offset]];
			for(int j= curNode.start_edgeid;
					j< curNode.start_edgeid + curNode.num_edges;
					j++){
				int neighbour = edges[j];
				int oldval = atomicCAS(&spath[neighbour], INT_MAX, level+1);
				if(oldval == INT_MAX){
					int pos = atomicAdd((int*)&nextPos, 1);
					pnext[pos] = neighbour;
				}
			}
			offset += blockDim.x;
		}
		__syncthreads();
		int *tmp = pcurr;
		pcurr = pnext;
		pnext = tmp;
		curWsize = nextPos;
		//printf("%d %d %d\n", bx, tx, nextPos);
	}while(curWsize<blockDim.x*2 && curWsize >0);

	// write next wave to global next wave
	volatile __shared__ int goffset;
	if(tx ==0){
		//goffset = atomicAdd(&nextWaveSize_d, curWsize);
		*nextWaveSize_d = curWsize;
	}
	int offset = tid;
	while(offset<curWsize){
		nextWave[offset+goffset] = pcurr[offset];
		offset += blockDim.x;
	}

}

__global__ void bfs_multiblocks(
		Node_t* nodes,
		Edge_t* edges,
		int* spath,
		int frontWaveSize,
		int* frontWave,
		int* nextWave,
		int* nextWaveSize_d){

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tid = bx*blockDim.x + tx;

	volatile __shared__ int nextPos;
	volatile __shared__ int goffset;
	__shared__ int counter;
	__shared__ int nextWave_s[2048];//for large graph, this may be spill, but
									// we can't allocate too much, it will limit the
									// number of active blocks on one SM

	/*if(tx==0 && nextWaveSize_d !=0)
		printf("%d %d %d\n", bx, tid, nextWaveSize_d);*/

	if(tx==0){
		nextPos = 0;
		counter =0;
	}
	__syncthreads();

	if(tid < frontWaveSize){
		int nodeid = frontWave[tid];
		Node_t curNode = nodes[nodeid];
		int level = spath[nodeid];
		for(int j= curNode.start_edgeid;
				j< curNode.start_edgeid + curNode.num_edges;
				j++){
			int neighbour = edges[j];
			int oldval = atomicCAS(&spath[neighbour], INT_MAX, level+1);
			if(oldval == INT_MAX){
				int pos = atomicAdd((int*)&nextPos, 1);

				/* the return statement inside will cause the following
				 * syncthreads() no effect, may be a bug with nvcc compiler
				 * */
				/*if(pos >=2048){
					printf("Error, local frontwave array too small !!!\n");
					//return;
				}*/
				nextWave_s[pos] = neighbour;
			}

		}

	}

	__syncthreads();

	int localWsize = nextPos;
	//printf("%d %d %d\n", bx, tx, nextPos);
	if(tx ==0){
		goffset = atomicAdd(nextWaveSize_d, localWsize);
		/*if(goffset+localWsize >=MAX_WAVE_SIZE){
			printf("Error, global frontwave array too small !!!\n");
			//return;
		}*/

	}
	__syncthreads();

	int offset = tx;
	while(offset<localWsize){
		nextWave[offset+goffset] = nextWave_s[offset];
		offset += blockDim.x;
	}

}




