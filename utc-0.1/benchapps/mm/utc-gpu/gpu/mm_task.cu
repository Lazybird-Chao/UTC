/*
 * mm_task.cc
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#include "Utc.h"
#include "UtcGpu.h"
#include "mm_task.h"
#include <iostream>

using namespace iUtc;

template<typename T>
thread_local int MatrixMulWorker<T>::start_row;

template<typename T>
thread_local int MatrixMulWorker<T>::local_numRows;

template<typename T>
void MatrixMulWorker<T>::initImpl(T *mA, T *mB, T *mC, int M, int N, int P, int blockSize){
	if(__localThreadId == 0){
		matrixA = mA;
		matrixB = mB;
		matrixC = mC;
		sizeM = M;
		sizeN = N;
		sizeP = P;
		gpuBlockSize = blockSize;
		TaskBcastBy<int, 0>(this, &sizeM, 1, 0);
		TaskBcastBy<int, 0>(this, &sizeN, 1, 0);
		TaskBcastBy<int, 0>(this, &sizeP, 1, 0);
		blockRows = sizeM / __numGroupProcesses;
		localBlockA = new T[blockRows*sizeN]; // block*N
		localBlockB = new T[sizeN*blockRows]; // N*block
		localBlockC = new T[sizeM*blockRows];
		//std::cout<<ERROR_LINE<<" "<<sizeM<<" "<<sizeN<<" "<<sizeP<<std::endl;
		/*sharedA.init(sizeM*sizeN);
		if(__processIdInGroup == 0)
			memcpy(sharedA.getPtr(), matrixA, sizeM*sizeN*sizeof(T));
		sharedC.init(sizeM*sizeP);*/

	}
	__fastIntraSync.wait();
	int rowsPerThread = blockRows/__numLocalThreads;
	if(__localThreadId<blockRows%__numLocalThreads){
		local_numRows = rowsPerThread+1;
		start_row = __localThreadId*(rowsPerThread+1);
	}
	else{
		local_numRows = rowsPerThread;
		start_row = __localThreadId*rowsPerThread + blockRows%__numLocalThreads;
	}
	inter_Barrier();
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

template<typename T>
void MatrixMulWorker<T>::runImpl(double runtime[][4]){
	//__fastIntraSync.wait();
	if(__globalThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;

	GpuData<T> gA(blockRows*sizeN);
	GpuData<T> gB(blockRows*sizeN);
	GpuData<T> gC(blockRows*blockRows);

	/*
	 * scatter block of B to other procs
	 */
	timer.start();
	timer0.start();
	TaskScatterBy<T, 0>(this, matrixB, sizeN * blockRows, localBlockB, sizeN*blockRows, 0);
	__fastIntraSync.wait();
	double commtime = timer.stop();

	timer.start();
	gB.putD(localBlockB);
	double copytime = timer.stop();

	double comptime = 0;
	for(int i = 0; i < sizeM / blockRows; i++){
		/*
		 * load one block of A
		 */

		/*
		 * if(__localThreadId == 0){
			if(__processIdInGroup == 0)
				sharedA.loadblock(localBlockA, i*blockRows*sizeN, blockRows*sizeN);
			else
				sharedA.rloadblock(0, localBlockA, i*blockRows*sizeN, blockRows*sizeN);
		}
		__fastIntraSync.wait();
		*/
		if(__globalThreadId ==0)
			memcpy(localBlockA, matrixA+i*blockRows*sizeN, blockRows*sizeN*sizeof(T));
		timer.start();
		TaskBcastBy<T, 0>(this, localBlockA, blockRows*sizeN, 0);
		__fastIntraSync.wait();
		commtime += timer.stop();

		timer.start();
		gA.putD(localBlockA);
		copytime += timer.stop();
		/*
		 * do local compute
		 */
		timer.start();
		int gridSizeX = (sizeP+gpuBlockSize-1) / gpuBlockSize;
		int gridSizeY = (sizeM+gpuBlockSize-1) / gpuBlockSize;
		dim3 grid(gridSizeX, gridSizeY, 1);
		dim3 block(gpuBlockSize, gpuBlockSize, 1);
		gpuMatrixMulKernel<T><<<grid, block, 0, __streamId>>>(
				gA.getD(),
				gB.getD(),
				gC.getD(true),
				sizeM,
				sizeN,
				sizeP,
				gpuBlockSize);
		checkCudaErr(cudaStreamSynchronize(__streamId));
		checkCudaErr(cudaGetLastError());
		comptime += timer.stop();

		timer.start();
		gC.fetch(localBlockC+i*blockRows*blockRows);
		copytime += timer.stop();
		/*
		 * store one block of C
		 */
		/*
		if(__localThreadId == 1){
			if(__processIdInGroup == 0)
				sharedC.storeblock(localBlockC+i*blockRows*blockRows,
						__processIdInGroup*blockRows*sizeN + i*blockRows*blockRows,
						blockRows*blockRows);
			else
				sharedC.rstoreblock(0, localBlockC+i*blockRows*blockRows,
						__processIdInGroup*blockRows*sizeN+i*blockRows*blockRows,
						blockRows*blockRows);
		}
		*/
		timer.start();
		TaskGatherBy<T, 0>(this, localBlockC+i*blockRows*blockRows, blockRows*blockRows,
								matrixC + i*blockRows*sizeN, blockRows*blockRows, 0);
		__fastIntraSync.wait();
		commtime += timer.stop();
	}
	inter_Barrier();
	double totaltime = timer0.stop();

	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = comptime;
		runtime[__localThreadId][2] = commtime;
		if(__localThreadId ==0){
			/*memcpy(matrixC, sharedC.getPtr(), sizeM*sizeP*sizeof(T));
			sharedC.destroy();
			sharedA.destroy();*/
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
		}
	}

}

template class MatrixMulWorker<float>;
template class MatrixMulWorker<double>;




















