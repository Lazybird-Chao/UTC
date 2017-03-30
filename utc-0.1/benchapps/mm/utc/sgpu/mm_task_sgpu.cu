/*
 * mm_task_sgpu.cu
 *
 *  Created on: Mar 21, 2017
 *      Author: chao
 */

#include "mm_task_sgpu.h"
#include "mm_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>


template<typename T>
void MatrixMulSGPU<T>::initImpl(T *mA, T *mB, T *mC, int M, int N, int P){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		matrixA = mA;
		matrixB = mB;
		matrixC = mC;
		sizeM = M;
		sizeN = N;
		sizeP = P;
	}
	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

template<typename T>
void MatrixMulSGPU<T>::runImpl(double *runtime, int blockSize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	GpuData<T> mA(sizeM*sizeN, memtype);
	GpuData<T> mB(sizeN*sizeP, memtype);
	GpuData<T> mC(sizeM*sizeP, memtype);
	mA.initH(matrixA);
	mB.initH(matrixB);

	/*
	 * copy data in
	 */
	timer.start();
	mA.sync();
	mB.sync();
	double copyinTime = timer.stop();

	/*
	 * call kernel
	 */
	timer.start();
	int gridSizeX = (sizeP+blockSize-1) / blockSize;
	int gridSizeY = (sizeM+blockSize-1) / blockSize;
	dim3 grid(gridSizeX, gridSizeY, 1);
	dim3 block(blockSize, blockSize, 1);
	gpuMatrixKernel<T><<<grid, block, 0,__streamId>>>(
				mA.getD(),
				mB.getD(),
				mC.getD(true),
				sizeM,
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
	mC.fetch(matrixC);
	//mC.fetch(matrixC);


	runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

template class MatrixMulSGPU<float>;
template class MatrixMulSGPU<double>;




