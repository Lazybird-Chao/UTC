/*
 * micro_task.h
 *
 *  Created on: Feb 23, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MICRO_UTC_MICRO_TASK_H_
#define BENCHAPPS_MICRO_UTC_MICRO_TASK_H_

#include "Utc.h"
#include "UtcGpu.h"

#include "cuda_runtime.h"
#include <iostream>

using namespace iUtc;

enum memtype_enum{
	pageable,
	pinmem,
	umem
};

template<typename T>
class microTest: public UserTaskBase{
private:
	int nscale;
	int blocksize;
	int nStreams;
	enum memtype_enum memtype;
	int loop;

	T* data;
	T* data_d;

	int nsize;
	int streamSize;

	cudaStream_t *streams;
	cudaStream_t mystream;

	void kernel_pageable(double* runtime);

	void kernel_pinned(double* runtime);

	void kernel_umem(double* runtime);

	T maxError(T* data, int n);

public:
	void initImpl(int nscale, int blocksize, int nStreams, int loop, enum memtype_enum memtype);

	void runImpl(double *runtime);

	~microTest();
};

#endif /* BENCHAPPS_MICRO_UTC_MICRO_TASK_H_ */
