/*
 * kmeans_task.h
 *
 *  Created on: Oct 13, 2017
 *      Author: chaoliu
 */

#ifndef KMEANS_TASK_H_
#define KMEANS_TASK_H_

#include "Utc.h"
#define MAX_TIMER 9

using namespace iUtc;

template<typename T>
class kmeansWorker: public UserTaskBase{
private:
	T* objects;
	int numObjs;
	int numCoords;
	int numClusters;
	T* clusters;
	int* clusters_size;
	int changedObjs;

	T* proc_clusters;
	int* proc_clusters_size;
	T* proc_clusters_array;
	int* proc_clusters_size_array;
	int proc_changedObjs;
	int* proc_changedObjs_array;
	static thread_local int thread_numObjs;
	static thread_local int thread_startObjIndex;

public:
	void initImpl(T* objects, T* clusters, int numObjs, int numCoords, int numClusters);

	void runImpl(double runtime[][MAX_TIMER], T threshold, int maxiters, int* loop);

	void cluster(int* membership, int* newClustersSize, T* newClusters);

};


#endif /* KMEANS_TASK_H_ */
