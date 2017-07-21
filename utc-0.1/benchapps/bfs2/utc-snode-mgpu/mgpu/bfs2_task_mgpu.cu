/*
 * bfs2_task_sgpu.cu
 *
 *  Created on: Mar 28, 2017
 *      Author: chaoliu
 */

#include "bfs2_task_mgpu.h"
#include "bfs2_kernel.h"
#include "../../../common/helper_err.h"


thread_local int bfsMGPU::local_numNodes;
thread_local int bfsMGPU::local_startNodeIndex;

void bfsMGPU::initImpl(Node_t *graph_nodes,
			Edge_t *graph_edges,
			int *shortestPath,
			int total_graph_nodes,
			int total_graph_edges,
			int src_node){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";

		this->graph_nodes = graph_nodes;
		this->graph_edges = graph_edges;
		this->shortestPath = shortestPath;
		this->total_graph_edges = total_graph_edges;
		this->total_graph_nodes = total_graph_nodes;
		this->src_node = src_node;

		this->g_spath = new int[total_graph_nodes];
		memcpy(this->g_spath, shortestPath, total_graph_nodes*sizeof(int));
		this->spath_array = new int[total_graph_nodes*__numLocalThreads];
		memset(this->spath_array, 0, total_graph_nodes*__numLocalThreads*sizeof(int));
		this->g_frontWave = new uint8_t[total_graph_nodes];
		memset(this->g_frontWave, 0, total_graph_nodes*sizeof(uint8_t));
		this->nextWave_array = new uint8_t[total_graph_nodes*__numLocalThreads];
		memset(this->nextWave_array, 0, total_graph_nodes*__numLocalThreads*sizeof(uint8_t));
		this->stopflag_array = new uint8_t[__numLocalThreads];
	}
	intra_Barrier();
	int nodesPerThread = total_graph_nodes / __numLocalThreads;
	if(__localThreadId<total_graph_nodes%__numLocalThreads){
		local_numNodes = nodesPerThread+1;
		local_startNodeIndex = __localThreadId*(nodesPerThread+1);
	}
	else{
		local_numNodes = nodesPerThread;
		local_startNodeIndex = __localThreadId*nodesPerThread + total_graph_nodes%__numLocalThreads;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}


void bfsMGPU::runImpl(double runtime[][4],
			int blocksize,
			int batch,
			MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	GpuData<uint8_t> graph_nextWaveFlag(total_graph_nodes, memtype);
	graph_nextWaveFlag.initH(0);
	GpuData<uint8_t> graph_frontWaveFlag(local_numNodes, memtype);
	//graph_frontWaveFlag.initH(0);
	stopflag_array[__localThreadId]=1;
	bool stopflag = false;
	if(src_node>=local_startNodeIndex &&
			src_node<local_startNodeIndex+local_numNodes){
		g_frontWave[src_node]=1;
		stopflag_array[__localThreadId]=0;
	}
	GpuData<Node_t> nodes(total_graph_nodes, memtype);
	GpuData<Edge_t> edges(total_graph_edges, memtype);
	GpuData<int> spath(total_graph_nodes, memtype);
	nodes.initH(graph_nodes);
	edges.initH(graph_edges);
	spath.initH(shortestPath);
	intra_Barrier();
	//std::cout<<total_graph_nodes<<" "<<total_graph_edges<<" "<<src_node<<std::endl;

	/*
	 * copyin
	 */
	timer0.start();
	double copyinTime = 0;
	timer.start();
	nodes.sync();
	edges.sync();
	//spath.sync();
	//graph_frontWaveFlag.sync();
	graph_nextWaveFlag.sync();
	copyinTime = timer.stop();


	/*
	 * do bfs search
	 */
	std::cout<<"start bfs processing ..."<<std::endl;
	double kernelTime=0;
	double copyoutTime=0;
	while(stopflag==false){
		stopflag = true;
		if(stopflag_array[__localThreadId]==0){
			stopflag_array[__localThreadId] =1;
			timer.start();
			graph_frontWaveFlag.initD(g_frontWave+local_startNodeIndex);
			spath.initD(g_spath);
			graph_nextWaveFlag.initD(0);
			copyinTime += timer.stop();


			timer.start();
			dim3 block(blocksize, 1, 1);
			dim3 grid((local_numNodes+blocksize*batch-1)/(blocksize*batch),1,1);
			bfs_findFront<<<grid, block, 0, __streamId>>>(
					nodes.getD(),
					edges.getD(),
					spath.getD(true),
					graph_frontWaveFlag.getD(true),
					graph_nextWaveFlag.getD(true),
					local_numNodes,
					batch,
					local_startNodeIndex);

			checkCudaErr(cudaGetLastError());
			checkCudaErr(cudaStreamSynchronize(__streamId));
			//checkCudaErr(cudaDeviceSynchronize());
			kernelTime += timer.stop();

			timer.start();
			graph_nextWaveFlag.fetchD(nextWave_array+__localThreadId*total_graph_nodes);
			spath.fetchD(spath_array+__localThreadId*total_graph_nodes);
			copyoutTime += timer.stop();
		}

		intra_Barrier();
		//std::cout<<local_startNodeIndex<<" "<<local_numNodes<<std::endl;
		for(int j=0; j<__numLocalThreads; j++){
		for(int i=local_startNodeIndex; i<local_startNodeIndex+local_numNodes; i++){
			//for(int j=0; j<__numLocalThreads; j++){
				if(nextWave_array[j*total_graph_nodes+i]==1){
					g_frontWave[i] = 1;
					g_spath[i] = spath_array[j*total_graph_nodes+i];
					if(stopflag_array[__localThreadId]==1){
						stopflag_array[__localThreadId]=0;
					}
					nextWave_array[j*total_graph_nodes+i] = 0;
				}
			}
		}
		intra_Barrier();
		for(int i=0; i<__numLocalThreads; i++){
			if(stopflag_array[i] == 0){
				stopflag = false;
				break;
			}
		}
		intra_Barrier();
		//std::cout<<__localThreadId<<": "<<stopflag<<std::endl;
		/*if(__localThreadId==0){
			std::cout<<__localThreadId<<": "<<(int)stopflag<<std::endl;
		}*/
	}

	totaltime = timer0.stop();


	runtime[__localThreadId][0] = totaltime;
	//runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[__localThreadId][1] = kernelTime;
	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;

	intra_Barrier();
	if(__localThreadId ==0){
		memcpy(shortestPath, g_spath, sizeof(int)*total_graph_nodes);
		delete g_spath;
		delete spath_array;
		delete nextWave_array;
		delete g_frontWave;
		delete stopflag_array;

		std::cout<<"task: "<<getCurrentTask()->getName()<<" thread "<<__localThreadId<<" finish runImpl.\n";
	}
}

