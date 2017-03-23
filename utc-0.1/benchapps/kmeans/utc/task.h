/*
 * task.h
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"

template<typename T>
class ClusterDataInit: public UserTaskBase{
public:
	void runImpl(int isBinaryFile,
			char* filename,
			T *objects,
			int *numObjs,
			int *numCoords);
};


template<typename T>
class Output: public UserTaskBase{
public:
	void runImpl(char* filename, T *clusters, int numClusters, int numCoords);
};



#endif /* TASK_H_ */
