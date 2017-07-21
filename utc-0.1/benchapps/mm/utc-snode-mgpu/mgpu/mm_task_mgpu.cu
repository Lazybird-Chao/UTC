/*
 * mm_task_sgpu.cu
 *
 *  Created on: Mar 21, 2017
 *      Author: chao
 */

#include "mm_task_mgpu.h"
#include "mm_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>

template<typename T>
thread_local int MatrixMulMGPU<T>::start_row;

template<typename T>
thread_local int MatrixMulMGPU<T>::local_numRows;

template<typename T>
void MatrixMulMGPU<T>::initImpl(T *mA, T *mB, T *mC, int M, int N, int P){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		matrixA = mA;
		matrixB = mB;
		matrixC = mC;
		sizeM = M;
		sizeN = N;
		sizeP = P;
	}
	int rowsPerThread = M/__numLocalThreads;
	if(__localThreadId<M%__numLocalThreads){
		local_numRows = rowsPerThread+1;
		start_row = __localThreadId*(rowsPerThread+1);
	}
	else{
		local_numRows = rowsPerThread;
		start_row = __localThreadId*rowsPerThread + M%__numLocalThreads;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

template<typename T>
void MatrixMulMGPU<T>::runImpl(double runtime[][4], int blockSize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;

	GpuData<T> partial_mA(local_numRows*sizeN, memtype);
	GpuData<T> mB(sizeN*sizeP, memtype);
	GpuData<T> mC(local_numRows*sizeP, memtype);
	partial_mA.initH(matrixA + (start_row*sizeN));
	mB.initH(matrixB);

	/*
	 * copy data in
	 */
	timer0.start();
	timer.start();
	partial_mA.sync();
	mB.sync();
	double copyinTime = timer.stop();

	/*
	 * call kernel
	 */
	timer.start();
	int gridSizeX = (sizeP+blockSize-1) / blockSize;
	int gridSizeY = (local_numRows+blockSize-1) / blockSize;
	dim3 grid(gridSizeX, gridSizeY, 1);
	dim3 block(blockSize, blockSize, 1);
	gpuMatrixKernel<T><<<grid, block, 0,__streamId>>>(
				partial_mA.getD(),
				mB.getD(),
				mC.getD(true),
				local_numRows,
				sizeN,
				sizeP,
				blockSize);
	checkCudaErr(cudaStreamSynchronize(__streamId));
	checkCudaErr(cudaGetLastError());
	double kernelTime = timer.stop();

	/*
	 * copy out
	 */
	timer.start();
	mC.sync();
	double copyoutTime = timer.stop();
	double totaltime = timer0.stop();
	mC.fetch(matrixC + (start_row*sizeP));
	intra_Barrier();


	//runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1] = kernelTime;
	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

template class MatrixMulMGPU<float>;
template class MatrixMulMGPU<double>;




