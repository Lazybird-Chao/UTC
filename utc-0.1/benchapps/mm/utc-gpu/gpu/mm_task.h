/*
 * mm_task.h
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MM_UTC_GPU_GPU_MM_TASK_H_
#define BENCHAPPS_MM_UTC_GPU_GPU_MM_TASK_H_

template<typename T>
class MatrixMulWorker : public UserTaskBase{
private:
	T *matrixA;
	T *matrixB;
	T *matrixC;
	int sizeM;
	int sizeN;
	int sizeP;

	int blockRows;
	T* localBlockA;
	T* localBlockB;
	T* localBlockC;

	int gpuBlockSize;

	//GlobalScopedData<T> sharedA;
	//GlobalScopedData<T> sharedC;

	static thread_local int start_row;
	static thread_local int local_numRows;

public:
	void initImpl(T *mA, T *mB, T *mC, int M, int N, int P, int blockSize);

	void runImpl(double runtime[][4]);
};



#endif /* BENCHAPPS_MM_UTC_GPU_GPU_MM_TASK_H_ */
