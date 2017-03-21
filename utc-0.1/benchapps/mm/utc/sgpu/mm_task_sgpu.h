/*
 * mm_task_sgpu.h
 *
 *  Created on: Mar 21, 2017
 *      Author: chao
 */

#ifndef MM_TASK_SGPU_H_
#define MM_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"

using namespace iUtc;


template<typename T>
class MatrixMulSGPU:public UserTaskBase{
private:
	T *matrixA;
	T *matrixB;
	T *matrixC;
	int sizeM;
	int sizeN;
	int sizeP;

public:
	void initImpl(T *mA, T *mB, T *mC, int M, int N, int P);

	void runImpl(double *runtime, int blocksize, MemType memtype);
};



#endif /* MM_TASK_SGPU_H_ */
