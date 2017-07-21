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

	static thread_local T* partial_obj_startPtr;
	static thread_local int partial_numObjs;
	T* objsNN_array;
	T* distance_array;
	static thread_local T* objsNN_array_startPtr;
	static thread_local T* distance_array_startPtr;

public:
	void initImpl(T* objects, T* objsNN, int numObjs, int numCoords, int numNN);

	void runImpl(double runtime[][6], MemType memtype);

};




#endif /* KMEANS_TASK_SGPU_H_ */
