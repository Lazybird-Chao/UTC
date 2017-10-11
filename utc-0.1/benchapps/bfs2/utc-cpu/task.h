/*
 * task.h
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"
#include "bfs2_comm_data.h"

class GraphInit: public UserTaskBase{
public:
	void runImpl(char* filepath,
				Node_t **graph_nodes,
				Edge_t **graph_edges,
				int *total_graph_nodes,
				int *total_graph_edges,
				int *source_nodeid);

};

class Output : public UserTaskBase{
public:
	void runImpl(char* filepath, int* spath, int total_nodes);
};


class bfsWorker : public UserTaskBase{
private:
	Node_t *graph_nodes;
	Edge_t *graph_edges;
	int *shortestPath;
	int total_graph_nodes;
	int total_graph_edges;
	int src_node;

	static thread_local int local_numNodes;
	static thread_local int local_startNodeIndex;
	bool *front_wave;
	bool *next_wave;
	bool *stopflag;

public:
	void initImpl(Node_t *graph_nodes,
			Edge_t *graph_edges,
			int *shortestPath,
			int total_graph_nodes,
			int total_graph_edges,
			int src_node);

	void runImpl(double runtime[][1]);
};


#endif /* TASK_H_ */
