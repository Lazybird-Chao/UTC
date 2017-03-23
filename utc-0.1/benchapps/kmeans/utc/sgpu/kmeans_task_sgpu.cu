/*
 * kmeans_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "kmeans_task_sgpu.h"
#include "Utc.h"

using namespace iUtc;

#define MAX_ITERATION 300


template<typename T>
void kmeansSGPU<T>::initImpl(T*objects, T*clusters, int numObjs, int numCoords, int numClusters){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->objects = objects;
		this->clusters = clusters;
		this->numClusters = numClusters;
		this->numObjs = numObjs;
		this->numCoords = numCoords;

	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void kmeansSGPU<T>::runImpl(double *runtime, T threshold, int* loopcounters, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	T *new_clusters = new T[numClusters*numCoords];
	T *new_clustersize = new int[numClusters];
	T *membership = new int[numObjs];
	/* Initialize membership, no object belongs to any cluster yet */
	for (int i = 0; i < numObjs; i++)
		membership[i] = -1;
	for(int i=0; i< numClusters*numCoords; i++)
		new_clusters[i] = 0;
	for(int i=0; i<numClusters; i++)
		new_clustersize[i]=0;


	GpuData<T> objs_d(numObjs*numCoords);
	GpuData<int> memship_d(numObjs);
	GpuData<T> clusters_d(numClusters*numCoords);

	timer.start();
	objs_d.init(objects);
	clusters_d.init(clusters);
	double copyinTime = timer.stop();

	std::cout<<"Start clustering..."<<std::endl;
	double kernelTime =0;
	double copyoutTime = 0;
	double hostCompTime = 0;

	int batchPerThread = 16;
	int blocksize = 256;
	int gridsize = (numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	int changedObjs =0;
	loopcounters = 0;
	do{
		timer.start();
		clusters_d.sync();
		copyinTime += timer.stop();

		timer.start();
		kmeans_kernel<<<grid, block, 0, __streamId>>>(
				objs_d.getD(),
				numCoords,
				numObjs,
				numClusters,
			    clusters_d.getD(),
			    memship_d.getD(true),
			    batchPerThread);
		cudaStreamSynchronize(__streamId);
		checkCudaErr(cudaGetLastError());
		kernelTime += timer.stop();

		timer.start();
		memship_d.sync();
		copyoutTime += timer.stop();

		/*
		 * compute new clusters
		 */
		timer.start();
		changedObjs = 0;
		int *new_membership = memship_d.getH();
		for(int i=0; i<numObjs; i++){
			if(new_membership[i] != membership[i]){
				changedObjs++;
				membership[i] = new_membership[i];
			}
			new_clustersize[membership[i]]++;
			for(int j=0; j<numCoords; j++)
				new_clusters[membership[i]*numCoords + j] += objects[i*numCoords+j];
		}
		//std::cout<<__LINE__<<std::endl;
		T* tmp_clusters = clusters_d.getH(true);
		for(int i=0; i<numClusters; i++){
			for(int j=0; j<numCoords; j++){
				if(new_clustersize[i]>1)
					tmp_clusters[i*numCoords +j] = new_clusters[i*numCoords+j]/new_clustersize[i];
				else
					tmp_clusters[i*numCoords +j] = new_clusters[i*numCoords+j];
				new_clusters[i*numCoords + j] = 0;
			}
			new_clustersize[i]=0;
		}
		hostCompTime += timer.stop();

	}while(loopcounters++ < MAX_ITERATION && (T)changedObjs/numObjs > threshold );
	clusters_d.fetch(clusters);

	delete new_clustersize;
	delete new_clusters;
	delete membership;

	runtime[0] = kernelTime + copyinTime + copyoutTime + hostCompTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
	runtime[4] = hostCompTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

template class kmeansSGPU<float>;
template class kmeansSGPU<double>;

