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
class kmeansSGPU: public UserTaskBase{
private:
	T* objects;
	int numObjs;
	int numCoords;
	int numClusters;
	T* clusters;

public:
	void initImpl(T* objects, T* clusters, int numObjs, int numCoords, int numClusters);

	void runImpl(double *runtime, T threshold, int* loop, MemType memtype);

};




#endif /* KMEANS_TASK_SGPU_H_ */
