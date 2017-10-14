/*
 * kmeans_task.cc
 *
 *  Created on: Oct 13, 2017
 *      Author: chaoliu
 */
#include "kmeans_task.h"

#define MAX_ITERATION 20

template<typename T>
__inline T euclid_dist_2(int numdims, T *coord1, T *coord2) {
    int i;
    T ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

template<typename T>
__inline int find_nearest_cluster(int numClusters, int numCoords, T *object, T *clusters) {
    int   index, i;
    T dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, &clusters[0]);
    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);

        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return index;
}


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

		this->proc_clusters_array = new T[this->numClusters*this->numCoords*__numLocalThreads];
		this->proc_clusters_size_array = new int[this->numClusters*__numLocalThreads];
		this->proc_changedObjs_array = new int[__numLocalThreads];
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
	double commtime = 0;

	T *new_clusters = proc_clusters_array + __localThreadId*numClusters*numCoords;
	int *new_clustersize = proc_clusters_size_array + __localThreadId*numClusters;
	int *partial_membership = new int[thread_numObjs];
	/* Initialize membership, no object belongs to any cluster yet */
	for (int i = 0; i < thread_numObjs; i++)
		partial_membership[i] = -1;
	for(int i=0; i< numClusters*numCoords; i++)
		new_clusters[i] = 0;
	for(int i=0; i<numClusters; i++)
		new_clustersize[i]=0;

	timer0.start();
	timer.start();
	/*
	 * bcast objects and initial clusters data to all nodes
	 */
	TaskBcastBy<T>(this, objects, numObjs*numCoords, 0);
	TaskBcastBy<T>(this, clusters, numClusters*numCoords, 0);
	__fastIntraSync.wait();
	commtime  = timer.stop();
	if(__globalThreadId == 0)
		std::cout<<"start clustering ...\n";
	int loop = 1;
	do{
		proc_changedObjs_array[__localThreadId] = 0;
		memset(new_clustersize, 0, numClusters*sizeof(int));
		memset(new_clusters, 0, numClusters*numCoords*sizeof(T));
		timer.start();
		cluster(partial_membership, new_clustersize, new_clusters);
		computetime += timer.stop();
		__fastIntraSync.wait();
		timer.start();
		if(__localThreadId == 0){
			memset(proc_clusters, 0, numClusters*numCoords*sizeof(T));
			memset(proc_clusters_size, 0, numClusters*sizeof(int));
			proc_changedObjs = 0;
			for(int k=0; k<__numLocalThreads; k++){
				for(int i=0; i<numClusters; i++){
					for(int j=0; j<numCoords; j++){
						proc_clusters[i*numCoords +j] += proc_clusters_array[k*numClusters*numCoords+i*numCoords+j];
						//proc_clusters_array[k*numClusters*numCoords+i*numCoords+j] = 0;
					}
					proc_clusters_size[i] += proc_clusters_size_array[k*numClusters+i];
					//proc_clusters_size_array[k*numClusters+i] = 0;
				}
				proc_changedObjs += proc_changedObjs_array[k];
			}
		}
		//computetime += timer.stop();
		//__fastIntraSync.wait();
		timer.start();
		TaskReduceSumBy<T>(this, proc_clusters, clusters, numClusters*numCoords, 0);
		TaskReduceSumBy<int>(this, proc_clusters_size, clusters_size, numClusters, 0);
		TaskReduceSumBy<int>(this, &proc_changedObjs, &changedObjs, 1, 0);
		//__fastIntraSync.wait();
		if(__globalThreadId == 0){
			for(int i=0; i<numClusters; i++){
				if(clusters_size[i]>1){
					for(int j=0; j<numCoords; j++)
						clusters[i*numCoords+j] /= clusters_size[i];
				}
			}
			//std::cout<<changedObjs<<std::endl;
		}
		//__fastIntraSync.wait();
		TaskBcastBy<int>(this, &changedObjs, 1, 0);
		TaskBcastBy<T>(this, clusters, numClusters*numCoords, 0);
		__fastIntraSync.wait();
		commtime += timer.stop();
	}while(loop++ < maxiters && ((T)changedObjs)/numObjs > threashold);
	inter_Barrier();
	totaltime = timer0.stop();

	delete partial_membership;
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = commtime;
		*loopcounters = loop;
	}
	if(__localThreadId == 0){
		delete proc_clusters_array;
		delete proc_clusters_size_array;
		delete proc_clusters;
		delete proc_clusters_size;
		delete proc_changedObjs_array;
		delete clusters_size;
	}
	if(__globalThreadId == 0)
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	return;
}

template<typename T>
void kmeansWorker<T>::cluster(int* membership, int *newClusterSize, T *newClusters){
	for(int i = thread_startObjIndex; i< thread_startObjIndex + thread_numObjs; i++){
		int index = find_nearest_cluster(numClusters,
										numCoords,
										&objects[i*numCoords],
										clusters);
		if(membership[i-thread_startObjIndex] != index){
			proc_changedObjs_array[__localThreadId]++;
			membership[i-thread_startObjIndex] = index;
		}
		newClusterSize[index]++;
		for(int j = 0; j<numCoords; j++)
			newClusters[index*numCoords + j] += objects[i*numCoords + j];
	}
	return;
}

template class kmeansWorker<float>;
template class kmeansWorker<double>;
















