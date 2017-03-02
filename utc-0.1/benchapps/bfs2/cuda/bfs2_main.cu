/*
 * bfs_main.cu
 *
 *  Created on: Feb 28, 2017
 *      Author: chao
 *
 * The gpu version of BFS algorithm.
 * Similar to the sequential version, but use one cuda thread to process one
 * graph node in the font-wave in every iteration. So for each iteration, the
 * number of cuda threads of gpu kernel may change.
 *
 * usage:
 * 		Compile with Makefile.
 * 		run as: ./a.out -v -i inputfile -o outputfile
 * 			-v: print time info
 * 			-i: the input graph data file path
 * 			-o: output file path
 *
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
#include "../../common/helper_err.h"
#include "bfs2_comm_data.h"
#include "bfs2_kernel.h"
#include "bfs2_main.h"


int main(int argc, char*argv[]){
	bool printTime = false;
	char* input_path = NULL;
	char* output_path=NULL;
	int blocksize = 256;
	int batch = 1;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vi:o:b:h:"))!=EOF){
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
		case 'b':
			blocksize = atoi(optarg);
			break;
		case 'h':
			batch = atoi(optarg);
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
	for(int i=0; i<total_graph_nodes; i++){
		shortestPath[i] = INT_MAX;
	}
	shortestPath[source_nodeid] = 0;

	/*
	 *  do bfs
	 */
	double runtime[4];
	bfs_process(graph_nodes, graph_edges, shortestPath, total_graph_nodes,
			total_graph_edges, source_nodeid, blocksize, batch, runtime);

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
		std::cout<<"\t\tsource node id: "<<source_nodeid<<std::endl;
		std::cout<<"\ttime info: "<<std::endl;
		std::cout<<"\t\tTotal time: "<<std::fixed<<std::setprecision(4)
					<<1000*runtime[0]<<"(ms)"<<std::endl;
		std::cout<<"\t\tkernel time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopyin time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopyout time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;
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

void bfs_process(
		Node_t *graph_nodes,
		Edge_t *graph_edges,
		int *shortestPath,
		int total_graph_nodes,
		int total_graph_edges,
		int src_node,
		int blocksize,
		int batch,
		double *runtime){

	uint8_t *graph_nextWaveFlag = new uint8_t[total_graph_nodes];
	for(int i=0; i<total_graph_nodes; i++)
		graph_nextWaveFlag[i] = 0;
	uint8_t *graph_frontWaveFlag = new uint8_t[total_graph_nodes];
	for(int i=0; i<total_graph_nodes; i++)
		graph_frontWaveFlag[i] = 0;
	graph_frontWaveFlag[src_node] = 1;
	double t1, t2;
	/*
	 * create gpu memory
	 */
	cudaSetDevice(0);
	Node_t *graph_nodes_d;
	Edge_t *graph_edges_d;
	int *shortestPath_d;
	checkCudaErr(cudaMalloc(&graph_nodes_d,
			total_graph_nodes*sizeof(Node_t)));
	checkCudaErr(cudaMalloc(&graph_edges_d,
			total_graph_edges*sizeof(Edge_t)));
	checkCudaErr(cudaMalloc(&shortestPath_d,
			total_graph_nodes*sizeof(int)));

	uint8_t* frontWave_d;
	uint8_t* nextWave_d;
	checkCudaErr(cudaMalloc(&frontWave_d,
			total_graph_nodes*sizeof(uint8_t)));
	checkCudaErr(cudaMalloc(&nextWave_d,
			total_graph_nodes*sizeof(uint8_t)));

	/*
	 * copyin data
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(graph_nodes_d,
			graph_nodes,
			total_graph_nodes*sizeof(Node_t),
			cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(graph_edges_d,
				graph_edges,
				total_graph_edges*sizeof(Edge_t),
				cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(shortestPath_d,
				shortestPath,
				total_graph_nodes*sizeof(int),
				cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(frontWave_d, graph_frontWaveFlag,
				total_graph_nodes*sizeof(uint8_t),
				cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(nextWave_d, graph_nextWaveFlag,
				total_graph_nodes*sizeof(uint8_t),
				cudaMemcpyHostToDevice));
	t2 = getTime();
	double copyinTime =0;
	copyinTime += t2-t1;

	/*
	 * call kernel to do bfs
	 */


	std::cout<<"start bfs processing ..."<<std::endl;

	uint8_t stopflag = 0;
	uint8_t *stopflag_d;
	checkCudaErr(cudaMalloc(&stopflag_d,
			sizeof(uint8_t)));
	double kernelTime=0;
	double copyoutTime=0;
	while(stopflag==0){
		stopflag = 1;
		t1=getTime();
		checkCudaErr(cudaMemcpy(stopflag_d, &stopflag, sizeof(uint8_t), cudaMemcpyHostToDevice));
		t2 = getTime();
		copyinTime += t2-t1;

		t1 = getTime();
		dim3 block(blocksize, 1, 1);
		dim3 grid((total_graph_nodes+blocksize*batch-1)/(blocksize*batch),1,1);
		bfs_findFront<<<grid, block>>>(
				graph_nodes_d,
				graph_edges_d,
				shortestPath_d,
				frontWave_d,
				nextWave_d,
				total_graph_nodes,
				batch);

		bfs_updateFront<<<grid, block>>>(
				stopflag_d,
				frontWave_d,
				nextWave_d,
				total_graph_nodes,
				batch);

		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaDeviceSynchronize());
		t2 = getTime();
		kernelTime += t2 -t1;

		t1= getTime();
		checkCudaErr(cudaMemcpy(&stopflag, stopflag_d, sizeof(uint8_t), cudaMemcpyDeviceToHost));
		t2 = getTime();
		copyoutTime += t2-t1;
	}


	/*
	 * copy result back
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(shortestPath, shortestPath_d,
			total_graph_nodes*sizeof(int),
			cudaMemcpyDeviceToHost));
	t2 = getTime();
	copyoutTime += t2-t1;

	cudaFree(graph_nodes_d);
	cudaFree(graph_edges_d);
	cudaFree(shortestPath_d);
	cudaFree(frontWave_d);
	cudaFree(nextWave_d);
	cudaFree(stopflag_d);

	runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
}

