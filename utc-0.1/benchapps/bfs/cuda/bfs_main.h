/*
 * bfs_main.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_MAIN_H_
#define BFS_MAIN_H_

#include "bfs_comm_data.h"

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


#endif /* BFS_MAIN_H_ */
