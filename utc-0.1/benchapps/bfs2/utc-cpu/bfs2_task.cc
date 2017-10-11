/*
 * bfs2_task.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */
#include "task.h"
#include "Utc.h"

thread_local int bfsWorker::local_numNodes;
thread_local int bfsWorker::local_startNodeIndex;

void bfsWorker::initImpl(Node_t *graph_nodes,
			Edge_t *graph_edges,
			int *shortestPath,
			int total_graph_nodes,
			int total_graph_edges,
			int src_node){
	if(__localThreadId == 0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";
		this->graph_nodes = graph_nodes;
		this->graph_edges = graph_edges;
		this->shortestPath = shortestPath;
		this->total_graph_edges = total_graph_edges;
		this->total_graph_nodes = total_graph_nodes;
		this->src_node = src_node;

		this->front_wave = new bool[total_graph_nodes];
		this->next_wave = new bool[total_graph_nodes];
		this->front_wave[src_node] = true;
		this->shortestPath[src_node] = 0;
		stopflag = new bool[__numLocalThreads];
	}
	__fastIntraSync.wait();
	stopflag[__localThreadId] = false;
	int nodesPerThread = total_graph_nodes / __numLocalThreads;
	if(__localThreadId<total_graph_nodes%__numLocalThreads){
		local_numNodes = nodesPerThread+1;
		local_startNodeIndex = __localThreadId*(nodesPerThread+1);
	}
	else{
		local_numNodes = nodesPerThread;
		local_startNodeIndex = __localThreadId*nodesPerThread + total_graph_nodes%__numLocalThreads;
	}
	__fastIntraSync.wait();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

void bfsWorker::runImpl(double runtime[][1]){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	timer.start();
	bool stop = stopflag[__localThreadId];
	while(!stop){
		stop = true;
		if(!stopflag[__localThreadId]){
		for(int i = local_startNodeIndex; i < local_startNodeIndex + local_numNodes; i++){
			if(front_wave[i]==true){
				front_wave[i] = false;
				int cur_node = i;
				int level = shortestPath[cur_node];
				for(int j = graph_nodes[cur_node].start_edgeid;
						j< graph_nodes[cur_node].start_edgeid + graph_nodes[cur_node].num_edges;
						j++){
					int neighbour = graph_edges[j];
					if(shortestPath[neighbour] == INT_MAX){
						shortestPath[neighbour] = level + 1;
						next_wave[neighbour] = true;
					}
				}
			}
		}
		}
		stopflag[__localThreadId] = true;
		__fastIntraSync.wait();
		for(int i = local_startNodeIndex; i < local_startNodeIndex + local_numNodes; i++){
			if(next_wave[i] == true){
				next_wave[i] = false;
				front_wave[i] = true;
				stopflag[__localThreadId] = false;
			}
		}
		__fastIntraSync.wait();
		for(int i = 0; i<__numLocalThreads; i++){
			if(stopflag[i]== false){
				stop = false;
				break;
			}
		}
		__fastIntraSync.wait();
	}
	double totaltime = timer.stop();

	runtime[__localThreadId][0] = totaltime;
	intra_Barrier();
	if(__localThreadId == 0){
		delete front_wave;
		delete next_wave;
		delete stopflag;

		std::cout<<"task: "<<getCurrentTask()->getName()<<" thread "<<__localThreadId<<" finish runImpl.\n";
	}
}



