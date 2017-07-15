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
class nnMGPU: public UserTaskBase{
private:
	T* objects;
	int numObjs;
	int numCoords;
	int numNN;
	T* objsNN;

	thread_local T* partial_obj_startPtr;
	thread_local int partial_numObjs;
	T* objsNN_array;
	T* distance_array;
	thread_local T* objsNN_array_startPtr;
	thread_local T* distance_array_startPtr;

public:
	void initImpl(T* objects, T* objsNN, int numObjs, int numCoords, int numNN);

	void runImpl(double **runtime, MemType memtype);

};




#endif /* KMEANS_TASK_SGPU_H_ */
