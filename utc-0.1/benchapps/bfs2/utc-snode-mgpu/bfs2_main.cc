#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "mgpu/bfs2_task_mgpu.h"
#include "bfs2_comm_data.h"

int main(int argc, char**argv){
	bool printTime = false;
	char* input_path = NULL;
	char* output_path=NULL;
	int blocksize = 256;
	int batch = 1;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vt:p:m:i:o:b:h:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 'm': mtype = atoi(optarg);
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
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

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
	Task<bfsMGPU> bfs(ProcList(nthreads, 0), TaskType::gpu_task);
	bfs.init(graph_nodes,
			graph_edges,
			shortestPath,
			total_graph_nodes,
			total_graph_edges,
			source_nodeid);
	double runtime_m[8][4];
	bfs.run(runtime_m, blocksize, batch, memtype);
	bfs.wait();
	double runtime[4]={0,0,0,0};
	for(int i=0; i<nthreads; i++)
		for(int j=0; j<4; j++)
			runtime[j]+= runtime_m[i][j];
	for(int j=0; j<4; j++)
		runtime[j] /= nthreads;


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

	for(int i=0; i<4; i++)
		runtime[i] *= 1000;
	print_time(4, runtime);

	return 0;
}




