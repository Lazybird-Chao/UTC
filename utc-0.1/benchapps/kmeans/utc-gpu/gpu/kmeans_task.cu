/*
 * kmeans_task.cc
 *
 *  Created on: Oct 13, 2017
 *      Author: chaoliu
 */
#include "kmeans_task.h"
#include "kmeans_kernel.h"
#include "../../../common/helper_err.h"

#define MAX_ITERATION 20


template<typename T> thread_local int kmeansWorker<T>::thread_numObjs;
template<typename T> thread_local int kmeansWorker<T>::thread_startObjIndex;

template<typename T>
void kmeansWorker<T>::initImpl(T*objects, T*clusters, int numObjs, int numCoords, int numClusters){
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
	}
	if(__localThreadId == 0){
		if(__processIdInGroup ==0){
			this->objects = objects;
			this->clusters = clusters;
			this->numClusters = numClusters;
			this->numObjs = numObjs;
			this->numCoords = numCoords;
		}
		TaskBcastBy<int>(this, &(this->numClusters), 1, 0);
		TaskBcastBy<int>(this, &(this->numObjs), 1, 0);
		TaskBcastBy<int>(this, &(this->numCoords), 1, 0);
		if(objects == nullptr)
			this->objects = new T[this->numObjs*this->numCoords];
		if(clusters == nullptr)
			this->clusters = new T[this->numClusters*this->numCoords];
		this->clusters_size = new int[this->numClusters];

		this->proc_clusters = new T[this->numClusters*this->numCoords];
		this->proc_clusters_size  = new int[this->numClusters];

	}
	__fastIntraSync.wait();
	int objsPerThread = this->numObjs/__numGlobalThreads;
	if(__globalThreadId < this->numObjs % __numGlobalThreads){
		thread_numObjs = objsPerThread +1;
		thread_startObjIndex = __globalThreadId*(objsPerThread+1);
	}
	else{
		thread_numObjs = objsPerThread;
		thread_startObjIndex = __globalThreadId*objsPerThread + this->numObjs % __numGlobalThreads;
	}

	inter_Barrier();
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void kmeansWorker<T>::runImpl(double runtime[][MAX_TIMER], T threashold, int maxiters, int *loopcounters){
	if(__globalThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime = 0;
	double computetime = 0;
	double kerneltime = 0;
	double commtime = 0;
	double copytime = 0;

	T *new_clusters = proc_clusters;
	int *new_clustersize = proc_clusters_size;
	int *partial_membership = new int[thread_numObjs];
	/* Initialize membership, no object belongs to any cluster yet */
	for (int i = 0; i < thread_numObjs; i++)
		partial_membership[i] = -1;
	for(int i=0; i< numClusters*numCoords; i++)
		new_clusters[i] = 0;
	for(int i=0; i<numClusters; i++)
		new_clustersize[i]=0;

	GpuData<T> objs_d(numObjs*numCoords);
	GpuData<int> partial_membership_d(thread_numObjs);
	GpuData<T> clusters_d(numClusters*numCoords);

	timer0.start();
	timer.start();
	/*
	 * bcast objects and initial clusters data to all nodes
	 */
	TaskBcastBy<T>(this, objects, numObjs*numCoords, 0);
	TaskBcastBy<T>(this, clusters, numClusters*numCoords, 0);
	__fastIntraSync.wait();
	commtime  = timer.stop();
	timer.start();
	objs_d.putD(objects);
	partial_membership_d.putD(partial_membership);
	copytime = timer.stop();

	if(__globalThreadId == 0)
		std::cout<<"start clustering ...\n";
	int loop = 1;
	int batchPerThread = 1;
	int blocksize = 256;
	int gridsize = (thread_numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	do{
		memset(new_clustersize, 0, numClusters*sizeof(int));
		memset(new_clusters, 0, numClusters*numCoords*sizeof(T));
		timer.start();
		clusters_d.putD(clusters);
		copytime += timer.stop();
		/*
		 * compute new membership
		 */
		timer.start();
		kmeans_kernel<<<grid, block, 0, __streamId>>>(
					objs_d.getD(),
					numCoords,
					numObjs,
					numClusters,
					clusters_d.getD(),
					partial_membership_d.getD(true),
					batchPerThread,
					thread_startObjIndex);
			cudaStreamSynchronize(__streamId);
			checkCudaErr(cudaGetLastError());
		kerneltime += timer.stop();
		computetime += timer.stop();
		/*
		 * compute new clusters on host
		 */
		timer.start();
		partial_membership_d.sync();
		copytime += timer.stop();
		timer.start();
		proc_changedObjs = 0;
		int *new_membership = partial_membership_d.getH();
		for(int i = 0; i< thread_numObjs; i++){
			if(new_membership[i] != partial_membership[i]){
				proc_changedObjs++;
				partial_membership[i] = new_membership[i];
			}
			new_clustersize[new_membership[i]]++;
			for(int j=0; j<numCoords; j++)
				new_clusters[new_membership[i]*numCoords + j] += objects[(i+thread_startObjIndex)*numCoords+j];
		}
		computetime += timer.stop();
		timer.start();
		TaskReduceSumBy<T>(this, proc_clusters, clusters, numClusters*numCoords, 0);
		TaskReduceSumBy<int>(this, proc_clusters_size, clusters_size, numClusters, 0);
		TaskReduceSumBy<int>(this, &proc_changedObjs, &changedObjs, 1, 0);
		if(__globalThreadId == 0){
			for(int i=0; i<numClusters; i++){
				if(clusters_size[i]>1){
					for(int j=0; j<numCoords; j++)
						clusters[i*numCoords+j] /= clusters_size[i];
				}
			}
			//std::cout<<changedObjs<<std::endl;
		}
		TaskBcastBy<int>(this, &changedObjs, 1, 0);
		TaskBcastBy<T>(this, clusters, numClusters*numCoords, 0);
		commtime += timer.stop();
	}while(loop++ < maxiters && ((T)changedObjs)/numObjs > threashold);
	inter_Barrier();
	totaltime = timer0.stop();

	delete partial_membership;
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = kerneltime;
		runtime[__localThreadId][3] = commtime;
		runtime[__localThreadId][4] = copytime;
		*loopcounters = loop;
	}
	if(__localThreadId == 0){
		delete proc_clusters;
		delete proc_clusters_size;
		delete clusters_size;
	}
	if(__globalThreadId == 0)
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	return;
}

template class kmeansWorker<float>;
template class kmeansWorker<double>;
















