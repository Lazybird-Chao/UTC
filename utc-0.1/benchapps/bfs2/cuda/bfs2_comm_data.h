/*
 * bfs_comm_data.h
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 */

#ifndef BFS_COMM_DATA_H_
#define BFS_COMM_DATA_H_

typedef struct{
	int start_edgeid;
	int num_edges;
} Node_t;

/*typedef struct{
	int dst_nodeid;
	int weight;   // here it is always 1
} Edge_t;*/
typedef int Edge_t; // for simple, just ignore the weight


#define WHITE 0
#define GRAY  1
#define BLACK 2



#endif /* BFS_COMM_DATA_H_ */
