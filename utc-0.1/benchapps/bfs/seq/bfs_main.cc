/*
 * bfs_main.cc
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 *
 * Using BFS algorithm to find the single source Shortest-path to
 * all nodes in a un-weighted graph.
 * Here we use the level-synchronous algorithm for BFS implementation,
 * another popular method is use SparseMatrix-vector multiply algorithm
 * to implement BFS.
 *
 * This is the sequential version.
 *
 * usage:
 * 		Compile with Makefile.
 * 		run as: ./a.out -v -i inputfile -o outputfile
 * 			-v: print time info
 * 			-i: the input graph data file path
 * 			-o: output file path
 *
 */

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <iostream>
#include <iomanip>
#include <vector>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "bfs_main.h"



int main(int argc, char*argv[]){
	bool printTime = false;
	char* input_path=NULL;
	char* output_path=NULL;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vi:o:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 'i':
			input_path = optarg;
			break;
		case 'o':
			output_path = optarg;
			break;
		case '?':
			break;
		default:
			break;
		}
	}
	if(input_path == NULL){
		std::cerr<<"Need input file path with -i !!!"<<std::endl;
		return 1;
	}

	/*
	 * read input file and initialize the graph data
	 */
	std::cout<<"Read graph data ..."<<std::endl;
	Node_t *graph_nodes;
	Edge_t *graph_edges;
	int total_graph_nodes;
	int total_graph_edges;
	int source_nodeid;
	initGraphFromFile(input_path, graph_nodes, graph_edges,
			total_graph_nodes, total_graph_edges, source_nodeid);
	//std::cout<<total_graph_nodes<<" "<<total_graph_edges<<std::endl;
	int *shortestPath = new int[total_graph_nodes];
	// no need for this color array
	//uint8_t nodesColor = new uint8_t[total_graph_nodes];
	for(int i=0; i<total_graph_nodes; i++){
		shortestPath[i] = INT_MAX;
		//nodeColor[i] = WHITE;
	}

	/*
	 * do bfs processing
	 */
	std::cout<<"start bfs ..."<<std::endl;
	double t1, t2;
	t1 = getTime();
	bfs(graph_nodes, graph_edges, source_nodeid, shortestPath);
	t2 = getTime();
	double runtime = t2-t1;

	/*
	 * write result
	 */
	if(output_path!=NULL){
		std::cout<<"write output ..."<<std::endl;
		writeOutput(output_path, shortestPath, total_graph_nodes);
	}

	delete graph_nodes;
	delete graph_edges;
	delete shortestPath;

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tgraph info:"<<std::endl;
		std::cout<<"\t\tnodes: "<<total_graph_nodes<<std::endl;
		std::cout<<"\t\tedges: "<<total_graph_edges<<std::endl;
		std::cout<<"\t\tsource node id: "<<source_nodeid;
		std::cout<<"\ttime info: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
	}
	return 0;

}

void initGraphFromFile(char* infile,
		Node_t *&nodes, Edge_t *&edges,
		int &total_nodes, int &total_edges, int &src_node){
	FILE *fp = fopen(infile, "r");
	if(!fp){
		std::cerr<<"Can't open input file !!!"<<std::endl;
		exit(1);
	}
	int x, y, count;
	count = fscanf(fp, "%d", &total_nodes);
	nodes = new Node_t[total_nodes];

	for(int i=0; i<total_nodes; i++){
		count =fscanf(fp, "%d", &x);
		count=fscanf(fp, "%d", &y);

		nodes[i].start_edgeid = x;
		nodes[i].num_edges = y;
	}

	count =fscanf(fp, "%d", &src_node);
	count =fscanf(fp, "%d", &total_edges);
	edges = new Edge_t[total_edges];
	for(int i=0; i<total_edges; i++){
		count=fscanf(fp, "%d", &x);
		count=fscanf(fp, "%d", &y);
		edges[i].dst_nodeid = x;
		edges[i].weight = y;
	}

	fclose(fp);

}

void writeOutput(
		char *outfile,
		int *spath,
		int total_nodes){
	FILE *fp = fopen(outfile, "w");
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

void bfs(
		Node_t *nodes,
		Edge_t *edges,
		int src_node,
		int *spath){
	std::vector<int> *frontWave = new std::vector<int>();
	std::vector<int> *nextWave = new std::vector<int>();
	frontWave->push_back(src_node);
	spath[src_node]=0;
	int cur_node;
	int level = 0;
	while(frontWave->size()!=0){
		for(int i=0; i<frontWave->size(); i++){
			cur_node = frontWave->at(i);
			for(int j = nodes[cur_node].start_edgeid;
					j< nodes[cur_node].start_edgeid + nodes[cur_node].num_edges;
					j++){
				int neighbour = edges[j].dst_nodeid;
				if(spath[neighbour] == INT_MAX){
					spath[neighbour] = level + 1;
					nextWave->push_back(neighbour);
				}
			}
		}
		level++;
		std::vector<int> *tmp = frontWave;
		frontWave = nextWave;
		nextWave = tmp;
		nextWave->clear();
	}
	delete frontWave;
	delete nextWave;

}

