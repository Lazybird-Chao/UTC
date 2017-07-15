/*
 * bfs2_task_sgpu.h
 *
 *  Created on: Mar 28, 2017
 *      Author: chaoliu
 */

#ifndef BFS2_TASK_SGPU_H_
#define BFS2_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"
#include "../bfs2_comm_data.h"

using namespace iUtc;

class bfsMGPU: public UserTaskBase{
private:
	Node_t *graph_nodes;
	Edge_t *graph_edges;
	int *shortestPath;
	int total_graph_nodes;
	int total_graph_edges;
	int src_node;

	thread_local int local_numNodes;
	thread_local int local_startNodeIndex;
	int *g_spath;
	int *spath_array;
	uint8_t *g_frontWave;
	uint8_t *nextWave_array;
	uint8_t *stopflag_array;
	uint8_t stopflag;

public:
	void initImpl(Node_t *graph_nodes,
			Edge_t *graph_edges,
			int *shortestPath,
			int total_graph_nodes,
			int total_graph_edges,
			int src_node);

	void runImpl(double *runtime,
			int blocksize,
			int batch,
			MemType memtype);

};



#endif /* BFS2_TASK_SGPU_H_ */
