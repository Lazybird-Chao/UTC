/*
 * task.h
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MM_UTC_CPU_TASK_H_
#define BENCHAPPS_MM_UTC_CPU_TASK_H_

#include "Utc.h"

template<typename T>
class RandomMatrixGen : public UserTaskBase{
public:
	void runImpl(T **matrix, int *h, int *w, const char* filename, bool isBinary);

};




template<typename T>
void toFile(T* matrix, int h, int w, const char* file, bool isBinary);


template<typename T>
void fromFile(T* &matrix, int& h, int& w, const char* file, bool isBinary);


#endif /* BENCHAPPS_MM_UTC_CPU_TASK_H_ */
