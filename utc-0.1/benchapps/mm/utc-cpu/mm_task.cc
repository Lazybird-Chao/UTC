/*
 * mm_task.cc
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#include "Utc.h"
#include "task.h"
#include <iostream>

using namespace iUtc;

template<typename T>
thread_local int MatrixMulWorker<T>::start_row;

template<typename T>
thread_local int MatrixMulWorker<T>::local_numRows;

template<typename T>
void MatrixMulWorker<T>::initImpl(T *mA, T *mB, T *mC, int M, int N, int P){
	if(__localThreadId == 0){
		matrixA = mA;
		matrixB = mB;
		matrixC = mC;
		sizeM = M;
		sizeN = N;
		sizeP = P;
		TaskBcastBy<int, 0>(this, &sizeM, 1, 0);
		TaskBcastBy<int, 0>(this, &sizeN, 1, 0);
		TaskBcastBy<int, 0>(this, &sizeP, 1, 0);
		blockRows = sizeM / __numGroupProcesses;
		localBlockA = new T[blockRows*sizeN]; // block*N
		localBlockB = new T[sizeN*blockRows]; // N*block
		localBlockC = new T[sizeM*blockRows];
		if(matrixA == nullptr)
			matrixA = new T[sizeM*sizeN];
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
void MatrixMulWorker<T>::runImpl(double runtime[][3]){
	//__fastIntraSync.wait();
	if(__globalThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;

	/*
	 * scatter block of B to other procs
	 */
	timer.start();
	timer0.start();
	TaskScatterBy<T, 0>(this, matrixB, sizeN * blockRows, localBlockB, sizeN*blockRows, 0);
	TaskBcastBy<T>(this, matrixA, sizeM*sizeN, 0);
	__fastIntraSync.wait();
	double commtime = timer.stop();

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
		/*
		if(__globalThreadId ==0)
			memcpy(localBlockA, matrixA+i*blockRows*sizeN, blockRows*sizeN*sizeof(T));
		timer.start();
		TaskBcastBy<T, 0>(this, localBlockA, blockRows*sizeN, 0);
		__fastIntraSync.wait();
		commtime += timer.stop();
		*/
		memcpy(localBlockA+start_row*sizeN,
				matrixA+i*blockRows*sizeN + start_row*sizeN, local_numRows*sizeN*sizeof(T));
		//__fastIntraSync.wait();
		/*
		 * do local compute
		 */
		timer.start();
		T *c_start = &localBlockC[i*blockRows*blockRows];
		for(int j = start_row; j < start_row+local_numRows; j++){
			for(int l = 0; l < blockRows; l++){
				T tmp = 0;
				for(int k = 0; k < sizeN; k++){
					tmp += localBlockA[j*sizeN + k] * localBlockB[l*sizeN + k];
				}
				c_start[j*blockRows + l] = tmp;
			}
		}
		__fastIntraSync.wait();
		/*if(__localThreadId == 0)
			std::cout<<ERROR_LINE<<std::endl;*/
		comptime += timer.stop();
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

	}
	/*if(__localThreadId==0){
		sharedC.quiet();
	}*/
	timer.start();
	TaskGatherBy<T, 0>(this, localBlockC, blockRows*sizeN,
							matrixC, blockRows*sizeN, 0);
	__fastIntraSync.wait();
	commtime += timer.stop();
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




















