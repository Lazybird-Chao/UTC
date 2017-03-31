/*
 * bfs2_task_sgpu.cu
 *
 *  Created on: Mar 28, 2017
 *      Author: chaoliu
 */

#include "bfs2_task_sgpu.h"
#include "bfs2_kernel.h"
#include "../../../common/helper_err.h"

void bfsSGPU::initImpl(Node_t *graph_nodes,
			Edge_t *graph_edges,
			int *shortestPath,
			int total_graph_nodes,
			int total_graph_edges,
			int src_node){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->graph_nodes = graph_nodes;
		this->graph_edges = graph_edges;
		this->shortestPath = shortestPath;
		this->total_graph_edges = total_graph_edges;
		this->total_graph_nodes = total_graph_nodes;
		this->src_node = src_node;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}


void bfsSGPU::runImpl(double *runtime,
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
	GpuData<uint8_t> graph_frontWaveFlag(total_graph_nodes, memtype);
	graph_frontWaveFlag.initH(0);
	graph_frontWaveFlag.put(src_node, 1);
	GpuData<Node_t> nodes(total_graph_nodes, memtype);
	GpuData<Edge_t> edges(total_graph_edges, memtype);
	GpuData<int> spath(total_graph_nodes, memtype);
	nodes.initH(graph_nodes);
	edges.initH(graph_edges);
	spath.initH(shortestPath);
	GpuData<uint8_t> stopflag(1, memtype);
	stopflag.put(0, 0);
	//std::cout<<total_graph_nodes<<" "<<total_graph_edges<<" "<<src_node<<std::endl;

	/*
	 * copyin
	 */
	timer0.start();
	double copyinTime = 0;
	timer.start();
	nodes.sync();
	edges.sync();
	spath.sync();
	graph_frontWaveFlag.sync();
	graph_nextWaveFlag.sync();
	copyinTime = timer.stop();


	/*
	 * do bfs search
	 */
	std::cout<<"start bfs processing ..."<<std::endl;
	double kernelTime=0;
	double copyoutTime=0;
	while(stopflag.at(0)==0){
		stopflag.put(0,1);
		timer.start();
		stopflag.sync();
		copyinTime += timer.stop();

		timer.start();
		dim3 block(blocksize, 1, 1);
		dim3 grid((total_graph_nodes+blocksize*batch-1)/(blocksize*batch),1,1);
		bfs_findFront<<<grid, block, 0, __streamId>>>(
				nodes.getD(),
				edges.getD(),
				spath.getD(true),
				graph_frontWaveFlag.getD(true),
				graph_nextWaveFlag.getD(true),
				total_graph_nodes,
				batch);

		bfs_updateFront<<<grid, block, 0, __streamId>>>(
				stopflag.getD(true),
				graph_frontWaveFlag.getD(true),
				graph_nextWaveFlag.getD(true),
				total_graph_nodes,
				batch);

		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaStreamSynchronize(__streamId));
		//checkCudaErr(cudaDeviceSynchronize());
		kernelTime += timer.stop();

		timer.start();
		stopflag.sync();
		copyoutTime += timer.stop();
	}

	timer.start();
	spath.sync();
	copyoutTime += timer.stop();
	totaltime = timer0.stop();
	spath.fetch(shortestPath);

	runtime[0] = totaltime;
	//runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

