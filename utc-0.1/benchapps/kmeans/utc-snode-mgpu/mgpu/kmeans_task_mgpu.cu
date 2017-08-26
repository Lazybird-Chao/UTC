/*
 * kmeans_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "kmeans_task_mgpu.h"
#include "kmeans_kernel.h"
#include "../../../common/helper_err.h"

#define MAX_ITERATION 20

template<typename T> thread_local int kmeansMGPU<T>::local_numObjs;
template<typename T> thread_local int kmeansMGPU<T>::local_startObjIndex;

template<typename T>
void kmeansMGPU<T>::initImpl(T*objects, T*clusters, int numObjs, int numCoords, int numClusters){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->objects = objects;
		this->clusters = clusters;
		this->numClusters = numClusters;
		this->numObjs = numObjs;
		this->numCoords = numCoords;

		this->local_clusters_array = new T[numClusters*numCoords*__numLocalThreads];
		this->local_clusters_size_array = new int[numClusters*__numLocalThreads];
		this->local_changedObjs_array = new int[__numLocalThreads];
		this->g_clusters = new T[numClusters*numCoords];
		this->g_clusters_size  = new int[numClusters];

	}
	intra_Barrier();
	int objsPerThread = numObjs/__numLocalThreads;
	if(__localThreadId < numObjs % __numLocalThreads){
		local_numObjs = objsPerThread +1;
		local_startObjIndex = __localThreadId*(objsPerThread+1);
	}
	else{
		local_numObjs = objsPerThread;
		local_startObjIndex = __localThreadId*objsPerThread + numObjs % __numLocalThreads;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void kmeansMGPU<T>::runImpl(double runtime[][5], T threshold, int* loopcounters, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	T *new_clusters = local_clusters_array + __localThreadId*numClusters*numCoords;
	int *new_clustersize = local_clusters_size_array + __localThreadId*numClusters;
	int *partial_membership = new int[local_numObjs];
	/* Initialize membership, no object belongs to any cluster yet */
	for (int i = 0; i < local_numObjs; i++)
		partial_membership[i] = -1;
	for(int i=0; i< numClusters*numCoords; i++)
		new_clusters[i] = 0;
	for(int i=0; i<numClusters; i++)
		new_clustersize[i]=0;


	GpuData<T> partial_objs_d(local_numObjs*numCoords);
	GpuData<int> partial_memship_d(local_numObjs);
	GpuData<T> clusters_d(numClusters*numCoords);
	partial_objs_d.initH(objects+local_startObjIndex*numCoords);
	clusters_d.initH(clusters);

	timer0.start();
	timer.start();
	partial_objs_d.sync();
	clusters_d.sync();
	double copyinTime = timer.stop();


	//std::cout<<"Start clustering..."<<std::endl;
	double kernelTime =0;
	double copyoutTime = 0;
	double hostCompTime = 0;

	int batchPerThread = 1;
	int blocksize = 256;
	int gridsize = (local_numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	int changedObjs =0;
	int loops = 0;
	do{
		timer.start();
		clusters_d.sync();
		copyinTime += timer.stop();

		timer.start();
		kmeans_kernel<<<grid, block, 0, __streamId>>>(
				partial_objs_d.getD(),
				numCoords,
				local_numObjs,
				numClusters,
			    clusters_d.getD(),
			    partial_memship_d.getD(true),
			    batchPerThread);
		cudaStreamSynchronize(__streamId);
		checkCudaErr(cudaGetLastError());
		kernelTime += timer.stop();

		timer.start();
		partial_memship_d.sync();
		copyoutTime += timer.stop();

		/*
		 * compute local new clusters
		 */
		timer.start();
		changedObjs = 0;
		int *new_membership = partial_memship_d.getH();
		for(int i=0; i<local_numObjs; i++){
			if(new_membership[i] != partial_membership[i]){
				changedObjs++;
				partial_membership[i] = new_membership[i];
			}
			new_clustersize[partial_membership[i]]++;
			for(int j=0; j<numCoords; j++)
				new_clusters[partial_membership[i]*numCoords + j] += objects[(i+local_startObjIndex)*numCoords+j];
		}
		local_changedObjs_array[__localThreadId] = changedObjs;
		//std::cout<<__LINE__<<std::endl;

		/*
		 * gather local new clusters of each thread
		 */
		intra_Barrier();
		if(getUniqueExecution()){
			for(int i=0; i<numClusters; i++){
				for(int j=0; j<numCoords; j++){
					g_clusters[i*numCoords +j] = local_clusters_array[i*numCoords+j];
					local_clusters_array[i*numCoords+j] =0;
				}
				g_clusters_size[i] = local_clusters_size_array[i];
				local_clusters_size_array[i]=0;
			}
			g_changedObjs = local_changedObjs_array[0];
			for(int k=1; k<__numLocalThreads; k++){
				for(int i=0; i<numClusters; i++){
					for(int j=0; j<numCoords; j++){
						g_clusters[i*numCoords +j] += local_clusters_array[k*numClusters*numCoords+i*numCoords+j];
						local_clusters_array[k*numClusters*numCoords+i*numCoords+j] =0;
					}
					g_clusters_size[i] += local_clusters_size_array[k*numClusters+i];
					local_clusters_size_array[k*numClusters+i]=0;
				}
				g_changedObjs += local_changedObjs_array[k];
			}
			for(int i=0; i<numClusters; i++){
				if(g_clusters_size[i]>1){
					for(int j=0; j<numCoords; j++)
						g_clusters[i*numCoords+j] /= g_clusters_size[i];
				}
			}
		}
		intra_Barrier();
		clusters_d.putH(g_clusters);


		hostCompTime += timer.stop();

	}while(loops++ < MAX_ITERATION && (T)g_changedObjs/numObjs > threshold );

	intra_Barrier();
	totaltime = timer0.stop();


	delete partial_membership;

	//runtime[0] = kernelTime + copyinTime + copyoutTime + hostCompTime;
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1] = kernelTime;
	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;
	runtime[__localThreadId][4] = hostCompTime;
	if(__localThreadId ==0){
		memcpy(clusters, g_clusters, numClusters*numCoords*sizeof(T));
		*loopcounters = loops;

		delete local_clusters_array;
		delete local_clusters_size_array;
		delete local_changedObjs_array;
		delete g_clusters;
		delete g_clusters_size;

		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

template class kmeansMGPU<float>;
template class kmeansMGPU<double>;

