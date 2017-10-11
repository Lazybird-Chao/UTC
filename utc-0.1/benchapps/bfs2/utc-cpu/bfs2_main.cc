/*
 * bfs2_main.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"

using namespace iUtc;

#include "task.h"
#include "bfs2_comm_data.h"

#define MAX_THREADS 64

int main(int argc, char**argv){
	bool printTime = false;
	char* input_path = NULL;
	char* output_path=NULL;

	int nthreads=1;
	int nprocess=1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	std::cout<<"UTC context initialized !"<<std::endl;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vt:p:i:o:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
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
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs || nprocess > 1){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	/*
	 * init graph data
	 */
	Node_t *graph_nodes;
	Edge_t *graph_edges;
	int total_graph_nodes;
	int total_graph_edges;
	int source_nodeid;
	Task<GraphInit> initGraph(ProcList(0));
	initGraph.run(input_path,
			&graph_nodes,
			&graph_edges,
			&total_graph_nodes,
			&total_graph_edges,
			&source_nodeid);
	initGraph.wait();
	int *shortestPath = new int[total_graph_nodes];
	for(int i=0; i<total_graph_nodes; i++){
		shortestPath[i] = INT_MAX;
	}
	shortestPath[source_nodeid] = 0;

	/*
	 * do bfs
	 */
	std::cout<<"Do bfs task ... "<<std::endl;
	Task<bfsWorker> bfs(ProcList(nthreads, 0));
	bfs.init(graph_nodes,
			graph_edges,
			shortestPath,
			total_graph_nodes,
			total_graph_edges,
			source_nodeid);
	double runtime_m[MAX_THREADS][1];
	bfs.run(runtime_m);
	bfs.wait();


	/*
	 * write result
	 */
	if(output_path!=NULL){
		Task<Output> fileout(ProcList(0));
		fileout.run(output_path, shortestPath, total_graph_nodes);
		fileout.wait();
	}

	if(graph_nodes)
		delete graph_nodes;
	if(graph_edges)
		delete graph_edges;
	if(shortestPath)
		delete shortestPath;


	double runtime = 0.0;
	for(int i=0; i<nthreads; i++)
		runtime += runtime_m[i][0];
	runtime /= nthreads;
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tgraph info:"<<std::endl;
		std::cout<<"\t\tnodes: "<<total_graph_nodes<<std::endl;
		std::cout<<"\t\tedges: "<<total_graph_edges<<std::endl;
		std::cout<<"\t\tsource node id: "<<source_nodeid<<std::endl;
		std::cout<<"\ttime info: "<<std::endl;
		std::cout<<"\t\tTotal time: "<<std::fixed<<std::setprecision(4)
					<<1000*runtime<<"(ms)"<<std::endl;
	}

	runtime *= 1000;
	print_time(1, &runtime);

	return 0;
}

