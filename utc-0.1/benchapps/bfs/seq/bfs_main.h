/*
 * bfs_main.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_MAIN_H_
#define BFS_MAIN_H_


typedef struct{
	int start_edgeid;
	int num_edges;
} Node_t;

typedef struct{
	int dst_nodeid;
	int weight;   // here it is always 1
} Edge_t;


#define WHITE 0
#define GRAY  1
#define BLACK 2

void initGraphFromFile(
		char* infile,
		Node_t *&nodes,
		Edge_t *&edges,
		int &total_nodes,
		int &total_edges,
		int &src_node);

void writeOutput(
		char *outfile,
		int *spath,
		int total_nodes);

void bfs(
		Node_t *nodes,
		Edge_t *edges,
		int src_node,
		int *spath);



#endif /* BFS_MAIN_H_ */
