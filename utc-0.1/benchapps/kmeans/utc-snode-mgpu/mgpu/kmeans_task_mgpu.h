/*
 * kmeans_task_sgpu.h
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#ifndef KMEANS_TASK_SGPU_H_
#define KMEANS_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"

using namespace iUtc;

template<typename T>
class kmeansMGPU: public UserTaskBase{
private:
	T* objects;
	int numObjs;
	int numCoords;
	int numClusters;
	T* clusters;

	T* g_clusters;
	int* g_clusters_size;
	T* local_clusters_array;
	int* local_clusters_size_array;
	int g_changedObjs;
	int* local_changedObjs_array;
	thread_local int local_numObjs;
	thread_local int local_startObjIndex;

public:
	void initImpl(T* objects, T* clusters, int numObjs, int numCoords, int numClusters);

	void runImpl(double **runtime, T threshold, int* loop, MemType memtype);

};




#endif /* KMEANS_TASK_SGPU_H_ */
