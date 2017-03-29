/*
 * task.h
 *
 *  Created on: Mar 28, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include "bfs2_comm_data.h"
#include "Utc.h"
using namespace iUtc;

class GraphInit: public UserTaskBase{
public:
	void runImpl(char* filepath,
			Node_t **graph_nodes,
			Edge_t **graph_edges,
			int *total_graph_nodes,
			int *total_graph_edges,
			int *source_nodeid){
		FILE *fp = fopen(filepath, "r");
		if(!fp){
			std::cerr<<"Can't open input file !!!"<<std::endl;
			exit(1);
		}
		int x, y, count;
		count = fscanf(fp, "%d", total_graph_nodes);
		Node_t *nodes = new Node_t[*total_graph_nodes];

		for(int i=0; i<*total_graph_nodes; i++){
			count =fscanf(fp, "%d", &x);
			count=fscanf(fp, "%d", &y);

			nodes[i].start_edgeid = x;
			nodes[i].num_edges = y;
		}

		count =fscanf(fp, "%d", source_nodeid);
		count =fscanf(fp, "%d", total_graph_edges);
		Edge_t *edges = new Edge_t[*total_graph_edges];
		for(int i=0; i<*total_graph_edges; i++){
			count=fscanf(fp, "%d", &x);
			count=fscanf(fp, "%d", &y);
			//edges[i].dst_nodeid = x;
			//edges[i].weight = y;
			edges[i] = x;
		}
		*graph_nodes = nodes;
		*graph_edges = edges;

		fclose(fp);
	}
};


class Output: public UserTaskBase{
public:
	void runImpl(char* filepath, int* spath, int total_nodes){
		FILE *fp = fopen(filepath, "w");
		if(!fp){
			std::cout<<"Cann't open the output file !!!"<<std::endl;
			exit(1);
		}
		fprintf(fp, "%d\n", total_nodes);
		for(int i=0; i<total_nodes; i++){
			fprintf(fp, "%d %d\n", i, spath[i]);
		}
		fclose(fp);
	}
};


#endif /* TASK_H_ */
