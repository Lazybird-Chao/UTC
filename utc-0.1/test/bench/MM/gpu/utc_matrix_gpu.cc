/*
 * utc_matrix_gpu.cc
 *
 *  Created on: Nov 5, 2016
 *      Author: chao
 *
 *
 *      utc-gpu version of matrix multiply
 *      Single-GPU version
 *
 */

#include "Utc.h"
#include "UtcGpu.h"
#include "../../helper_getopt.h"
#include "../../helper_printtime.h"

#include "utc_matrix_gpu_kernel.h"

#include <iostream>

using namespace iUtc;

template <typename T>
//#define T float
class GpuMatrixMultiply: public UserTaskBase{
private:
	int matrixSize;
	int blockSize;

	T *matrixA;
	T *matrixB;
	T *matrixC;

	T *matrixA_d;
	T *matrixB_d;
	T *matrixC_d;

public:
	void initImpl(int matrixSize, int blockSize){
		if(__localThreadId ==0){
			std::cout<<"begin init ...\n";
			this->matrixSize = matrixSize;
			this->blockSize = blockSize;

			matrixA = (T*)malloc(sizeof(T)*matrixSize*matrixSize);
			matrixB = (T*)malloc(sizeof(T)*matrixSize*matrixSize);
			matrixC = (T*)malloc(sizeof(T)*matrixSize*matrixSize);

			for(int i=0; i<matrixSize; i++)
				for(int j=0; j<matrixSize; j++){
					matrixA[i*matrixSize + j] = (j + 1.0)/matrixSize;
					matrixB[i*matrixSize + j] = (j + 1.0)/matrixSize;
				}
		}
		intra_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
		}
	}

	void runImpl(double *runtime){
		std::cout<<"begin run ..."<<std::endl;
		Timer timer;
		Timer t2;

		//t2.start();
		timer.start();
		checkCudaRuntimeErrors(cudaMalloc(&matrixA_d, matrixSize*matrixSize*sizeof(T)));
		checkCudaRuntimeErrors(cudaMalloc(&matrixB_d, matrixSize*matrixSize*sizeof(T)));
		checkCudaRuntimeErrors(cudaMalloc(&matrixC_d, matrixSize*matrixSize*sizeof(T)));
		runtime[0] = timer.stop();

		t2.start();
		timer.start();
		checkCudaRuntimeErrors(cudaMemcpy(matrixA_d, matrixA, matrixSize*matrixSize*sizeof(T), cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(cudaMemcpy(matrixB_d, matrixA, matrixSize*matrixSize*sizeof(T), cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(cudaMemset(matrixC_d, 0, matrixSize*matrixSize*sizeof(T)));
		runtime[1] = timer.stop();

		GpuKernel mykernel;
		mykernel.setGridDim(matrixSize/blockSize, matrixSize/blockSize);
		mykernel.setBlockDim(blockSize, blockSize);
		mykernel.setNumArgs(5);
		mykernel.setArgs<T*>(0, matrixA_d);
		mykernel.setArgs<T*>(1, matrixB_d);
		mykernel.setArgs<T*>(2, matrixC_d);
		mykernel.setArgs<int>(3, matrixSize);
		mykernel.setArgs<int>(4, blockSize);
		timer.start();
		mykernel.launchKernel((const void*)&gpuMatrixKernel);
		runtime[2] = timer.stop();

		timer.start();
		checkCudaRuntimeErrors(cudaMemcpy(matrixC, matrixC_d, matrixSize*matrixSize*sizeof(T), cudaMemcpyDeviceToHost));
		runtime[3] = timer.stop();
		runtime[4] = t2.stop();

		long err = 0;
		T *res = (T*)malloc(sizeof(T)*matrixSize*matrixSize);
		err = compareCompute(res, 1e-10);

		if(err>0)
			std::cout<<"run error: "<<err<<std::endl;
		else
			std::cout<<"run correct!"<<std::endl;
		free(res);


	}

	long compareCompute(T* res, T eps=1e-6){
		long err=0;
		for(int i=0; i<matrixSize; i++){
			for(int j=0; j<matrixSize; j++){
				T tmp = 0;
				for(int k=0; k<matrixSize; k++)
					tmp +=matrixA[i*matrixSize +k] * matrixB[k*matrixSize +j];
				res[i*matrixSize + j] = tmp;
			}
		}
		for(int i=0; i<matrixSize*matrixSize; i++){
			if(fabs(matrixC[i] - res[i])/matrixSize > eps){
				std::cout<<matrixC[i] - res[i]<<std::endl;
				err++;
			}
		}
		return err;
	}

	~GpuMatrixMultiply(){
		if(matrixA){
		free(matrixA);
		free(matrixB);
		free(matrixC);
		}
		if(matrixA_d){
		cudaFree(matrixA_d);
		cudaFree(matrixB_d);
		cudaFree(matrixC_d);
		}
	}
};


int main(int argc, char **argv){
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads;
	int nprocs;
	int matrixSize = 1024;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:s:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:s:");
	}
	int procs = ctx.numProcs();
	if(nprocs != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	if(nthreads != 1){
		std::cerr<<"this program only allows one thread task!!!\n";
		return 1;
	}
	int myproc = ctx.getProcRank();
	double runtime[5];

	Task<GpuMatrixMultiply<double>> myMM(ProcList(0), TaskType::gpu_task);  // 1 thread on proc 0
	myMM.init(matrixSize, 32);

	myMM.run(runtime);

	myMM.wait();
	myMM.finish();

	if(myproc==0){
		std::cout<<"total time: "<<runtime[4]<<std::endl;
		std::cout<<"mem alloc time: "<<runtime[0]<<std::endl;
		std::cout<<"mem copyin time: "<<runtime[1]<<std::endl;
		std::cout<<"kernel run time: "<<runtime[2]<<std::endl;
		std::cout<<"mem copyout time: "<<runtime[3]<<std::endl;

		print_time(5, runtime);
	}


	return 0;

}
