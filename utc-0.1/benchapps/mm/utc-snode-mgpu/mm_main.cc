/*
 * mm_main.cc
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: /a.out -v -s 100
 * 		-v: print time info
 * 		-t: number of threads
 * 		-p: number of processes(nodes)
 * 		-m: gpu memtype
 * 		-s: the size of matrix, we assume a square matrix
 * 		-b: cuda block size, should able to divide matrix size
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "mgpu/mm_task_mgpu.h"


#define FTYPE float

int main(int argc, char**argv){
	bool printTime = false;
	int blockSize = 16;
	int matrixSize = 1024;
	char *fileout = nullptr;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	std::cout<<"UTC context initialized !\n";

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vt:p:m:s:b:o:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 'm': mtype = atoi(optarg);
			  break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case 'b':
			blockSize = atoi(optarg);
			break;
		case 'o':
			fileout = optarg;
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vt:p:m:s:b:o:");
	}
	if(matrixSize<=0){
		std::cerr<<"matrix size is illegal"<<std::endl;
		exit(1);
	}

	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

	/*
	 * create matrix and initialize with random number
	 */
	FTYPE *matrixA = new FTYPE[matrixSize*matrixSize];
	FTYPE *matrixB = new FTYPE[matrixSize*matrixSize];
	//FTYPE *matrixC = new FTYPE[matrixSize*matrixSize];
	Task<RandomMatrix<FTYPE>> matrixInit(ProcList(0));
	//matrixInit.run(matrixA, matrixSize, matrixSize, nullptr, false);
	//matrixInit.run(matrixB, matrixSize, matrixSize, nullptr, false);
	matrixInit.run(matrixA, matrixSize, matrixSize, "../input/16k_16k_A.txt", true);
	matrixInit.run(matrixB, matrixSize, matrixSize, "../input/16k_16k_B.txt", true);
	matrixInit.wait();

	//RandomMatrix<FTYPE>::toFile(matrixA, matrixSize, matrixSize, "4k_4k_A.txt", true);
	//RandomMatrix<FTYPE>::toFile(matrixB, matrixSize, matrixSize, "4k_4k_B.txt", true);

	//
	int increase = 8;
	RandomMatrix<FTYPE>::increaseRowBy(increase, matrixA, matrixSize, matrixSize);
	FTYPE *matrixC = new FTYPE[matrixSize*increase*matrixSize];

	/*
	 * do computation
	 */
	double runtime_m[8][4];
	Task<MatrixMulMGPU<FTYPE>> mm(ProcList(nthreads, 0), TaskType::gpu_task);
	mm.init(matrixA, matrixB, matrixC, matrixSize*increase, matrixSize, matrixSize);
	mm.run(runtime_m, blockSize, memtype);
	mm.wait();
	double runtime[4]={0,0,0,0};
	for(int i=0; i<nthreads; i++)
		for(int j=0; j<4; j++)
			runtime[j]+= runtime_m[i][j];
	for(int j=0; j<4; j++)
		runtime[j] /= nthreads;

	/*
	int error =0;
	for(int i=0; i<matrixSize; i++){
		for(int j=0; j<matrixSize; j++){
			FTYPE tmp = 0;
			for(int k=0; k<matrixSize; k++){
				tmp += matrixA[i*matrixSize+k]*matrixB[k*matrixSize + j];
			}
			if(fabs((tmp-matrixC[i*matrixSize +j])/tmp)>0.00001){
				error++;
				//std::cout<<tmp<<"  "<<matrixC[i*matrixSize +j]<<std::endl;
			}
		}
	}
	if(error == 0)
		std::cout<<"no errors in results\n";
	else
		std::cout<<"errors: "<<error<<std::endl;
	*/

	if(fileout){
		RandomMatrix<FTYPE>::toFile(matrixC, matrixSize*increase, matrixSize, fileout, true);
	}
	delete matrixA;
	delete matrixB;
	delete matrixC;
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tMatrix size: "<<matrixSize*increase<<" X "<<matrixSize<<std::endl;
		std::cout<<"\tcuda Block size: "<<blockSize<<std::endl;
		std::cout<<"\tMemtype: "<<mtype<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		//std::cout<<"\t\tmemory create time: "<<memcreateTime<<" s"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<runtime[3]<<"(s)"<<std::endl;

	}

	for(int i=0; i<4; i++)
		runtime[i] *= 1000;
	print_time(4, runtime);

	return 0;

}







