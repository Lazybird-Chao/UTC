/*
 * bfs_main.cu
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
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
#include "../../common/helper_err.h"
#include "bfs_main.h"
#include "bfs_comm_data.h"
#include "bfs_kernel.h"


int main(int argc, char*argv[]){
	bool printTime = false;
	char* input_path = NULL;
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
	int *shortestPath = new int[total_graph_nodes];
	for(int i=0; i<total_graph_nodes; i++){
		shortestPath[i] = INT_MAX;
	}


	/*
	 * create gpu memory
	 */
	Node_t *graph_nodes_d;
	Edge_t *graph_edges_d;
	int *shortestPath_d;
	checkCudaErr(cudaMalloc(&graph_nodes_d,
			total_graph_nodes*sizeof(Node_t)));
	checkCudaErr(cudaMalloc(&graph_edges_d,
			total_graph_edges*sizeof(Edge_t)));
	checkCudaErr(cudaMalloc(&shortestPath_d,
			total_graph_nodes*sizeof(Node_t)));

	/*
	 * copyin data
	 */
	checkCudaErr(cudaMemcpy(graph_nodes_d,
			graph_nodes,
			total_graph_nodes*sizeof(Node_t),
			cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(graph_edges_d,
				graph_nodes,
				total_graph_edges*sizeof(Edge_t),
				cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(shortestPath_d,
				shortestPath,
				total_graph_nodes*sizeof(int),
				cudaMemcpyHostToDevice));

	/*
	 * call kernel to do bfs
	 */
#define MAX_THREAD_PER_BLOCK 256
#define MAX_WAVE_SIZE	(total_graph_nodes/1000)
	int* frontWave_d;
	int* nextWave_d;
	// allocal wave array, and assume the real wave size will not exceed
	// MAX_WAVE_SIZE during the iteration
	checkCudaErr(cudaMalloc(&frontWave_d,
			MAX_WAVE_SIZE*sizeof(int)));
	checkCudaErr(cudaMalloc(&nextWave_d,
			MAX_WAVE_SIZE*sizeof(int)));
	int frontWaveSize;

	//add source node id to frontwave to start
	checkCudaErr(cudaMemcpy(frontWave_d, &source_nodeid,
			sizeof(int), cudaMemcpyHostToDevice));
	frontWaveSize = 1;

	while(frontWaveSize >0){
		/*checkCudaErr(cudaMemcpyToSymbol(&nextWaveSize, 0,
				sizeof(int), 0, cudaMemcpyHostToDevice));*/
		if(frontWaveSize > MAX_THREAD_PER_BLOCK){
			dim3 block(MAX_THREAD_PER_BLOCK, 1 ,1);
			dim3 grid((frontWaveSize+MAX_THREAD_PER_BLOCK-1)/MAX_THREAD_PER_BLOCK,1,1);
			bfs_multiblocks<<<grid, block>>>(
					graph_nodes_d,
					graph_edges_d,
					shortestPath_d,
					frontWaveSize,
					frontWave_d,
					nextWave_d);
		}
		else{
			dim3 block(MAX_THREAD_PER_BLOCK,1,1);
			dim3 grid(1,1,1);
			bfs_singleblock<<<grid, block>>>(
					graph_nodes_d,
					graph_edges_d,
					shortestPath_d,
					frontWaveSize,
					frontWave_d,
					nextWave_d);
		}
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaDeviceSynchronize());

		checkCudaErr(cudaMemcpyFromSymbol(&frontWaveSize, &nextWaveSize_d,
				sizeof(int), 0, cudaMemcpyDeviceToHost));
		int *tmp = frontWave_d;
		frontWave_d = nextWave_d;
		nextWave_d = tmp;
	}


	/*
	 * copy result back
	 */
	checkCudaErr(cudaMemcpy(shortestPath, shortestPath_d,
			total_graph_nodes*sizeof(int),
			cudaMemcpyDeviceToHost));

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
	cudaFree(graph_nodes_d);
	cudaFree(graph_edges_d);
	cudaFree(shortestPath_d);
	cudaFree(frontWave_d);
	cudaFree(nextWave_d);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tgraph info:"<<std::endl;
		std::cout<<"\t\tnodes: "<<total_graph_nodes<<std::endl;
		std::cout<<"\t\tedges: "<<total_graph_edges<<std::endl;
		std::cout<<"\t\tsource node id: "<<source_nodeid;
		//std::cout<<"\ttime info: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
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
		//edges[i].dst_nodeid = x;
		//edges[i].weight = y;
		edges[i] = x;
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


