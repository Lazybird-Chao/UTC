/*
 * mm_main.cc
 *
 * The single GPU matrix multiply program. Use single cuda stream and explicit
 * gpu memory.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: /a.out -v -s 100
 * 		-v: print time info
 * 		-s: the size of matrix, we assume a square matrix
 * 		-b: cuda block size, should able to divide matrix size
 *
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"
#include "mm_kernel.h"

#define FTYPE float


int main(int argc, char **argv){
	int matrixSize = 1024;
	bool printTime = false;
	int blockSize = 16;


	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vs:b:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case 'b':
			blockSize = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vs:b:");
	}
	if(matrixSize<=0){
		std::cerr<<"matrix size is illegal"<<std::endl;
		exit(1);
	}
	/*if(matrixSize%blockSize != 0){
		std::cerr<<"matrix size should be multiply of block size"<<std::endl;
		exit(1);
	}*/

	/*
	 * create matrix and initialize with random number
	 */
	FTYPE *matrixA = (FTYPE*)malloc(sizeof(FTYPE)*matrixSize*matrixSize);
	FTYPE *matrixB = (FTYPE*)malloc(sizeof(FTYPE)*matrixSize*matrixSize);
	FTYPE *matrixC = (FTYPE*)malloc(sizeof(FTYPE)*matrixSize*matrixSize);

	FTYPE rnumber = (FTYPE)(rand()%100)/(rand()%10);
	for(int i=0; i<matrixSize; i++)
		for(int j=0; j<matrixSize; j++){
			matrixA[i*matrixSize + j] = (j + rnumber)/matrixSize;
			matrixB[i*matrixSize + j] = (j - rnumber)/matrixSize;
		}

	cudaSetDevice(0);

	/*
	 * create gpu memory
	 */
	FTYPE *matrixA_d;
	FTYPE *matrixB_d;
	FTYPE *matrixC_d;
	double t1, t2;
	t1 = getTime();
	checkCudaErr(cudaMalloc(&matrixA_d, matrixSize*matrixSize*sizeof(FTYPE)));
	checkCudaErr(cudaMalloc(&matrixB_d, matrixSize*matrixSize*sizeof(FTYPE)));
	checkCudaErr(cudaMalloc(&matrixC_d, matrixSize*matrixSize*sizeof(FTYPE)));
	t2 = getTime();
	double memcreateTime = t2-t1;

	/*
	 * copy data in
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(matrixA_d, matrixA, matrixSize*matrixSize*sizeof(FTYPE), cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemcpy(matrixB_d, matrixB, matrixSize*matrixSize*sizeof(FTYPE), cudaMemcpyHostToDevice));
	checkCudaErr(cudaMemset(matrixC_d, 0, matrixSize*matrixSize*sizeof(FTYPE)));
	t2 = getTime();
	double copyinTime = t2-t1;

	/*
	 * call kernel
	 */
	int gridSize = (matrixSize+blockSize-1) / blockSize;
	dim3 grid(gridSize, gridSize, 1);
	dim3 block(blockSize, blockSize, 1);
	/*void* kernel_args[5];
	kernel_args[0] = (void*)&matrixA_d;
	kernel_args[1] = (void*)&matrixB_d;
	kernel_args[2] = (void*)&matrixC_d;
	kernel_args[3] = (void*)&matrixSize;
	kernel_args[4] = (void*)&blockSize;
	t1 = getTime();
	cudaLaunchKernel((void*)&gpuMatrixKernel, grid, block, (void**)kernel_args, 0, 0);
	*/
	gpuMatrixKernel<<<grid, block>>>(
			matrixA_d,
			matrixB_d,
			matrixC_d,
			matrixSize,
			matrixSize,
			matrixSize,
			blockSize);
	cudaDeviceSynchronize();
	checkCudaErr(cudaGetLastError());
	t2 = getTime();
	double kernelTime = t2 - t1;

	/*
	 * copy result out
	 */
	t1 = getTime();
	checkCudaErr(cudaMemcpy(matrixC, matrixC_d, matrixSize*matrixSize*sizeof(FTYPE), cudaMemcpyDeviceToHost));
	t2 = getTime();
	double copyoutTime = t2 - t1;

	/*for(int i=0; i<matrixSize; i++){
		for(int j=0; j<matrixSize; j++){
			FTYPE tmp = 0;
			for(int k=0; k<matrixSize; k++){
				tmp += matrixA[i*matrixSize+k]*matrixB[k*matrixSize + j];
			}
			if(fabs((tmp-matrixC[i*matrixSize +j])/tmp)>0.00001)
				std::cout<<tmp<<"  "<<matrixC[i*matrixSize +j]<<std::endl;
		}
	}*/

	free(matrixA);
	free(matrixB);
	free(matrixC);
	cudaFree(matrixA_d);
	cudaFree(matrixB_d);
	cudaFree(matrixC_d);
	cudaDeviceReset();

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tMatrix size: "<<matrixSize<<" X "<<matrixSize<<std::endl;
		std::cout<<"\tcuda Block size: "<<blockSize<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		//std::cout<<"\t\tmemory create time: "<<memcreateTime<<" s"<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<copyinTime<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<copyoutTime<<"(s)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<kernelTime<<"(s)"<<std::endl;

	}

}






