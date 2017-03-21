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
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "sgpu/mm_task_sgpu.h"


#define FTYPE float

int main(int argc, char**argv){
	bool printTime = false;
	int blockSize = 16;
	int matrixSize = 1024;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;
	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vt:p:m:s:b:");
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
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vt:p:m:s:b:");
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
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}

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
	FTYPE *matrixA = new FTYPE[sizeof(FTYPE)*matrixSize*matrixSize];
	FTYPE *matrixB = new FTYPE[sizeof(FTYPE)*matrixSize*matrixSize];
	FTYPE *matrixC = new FTYPE[sizeof(FTYPE)*matrixSize*matrixSize];
	Task<RandomMatrix<FTYPE>> matrixInit(ProcList(0));
	matrixInit.run(matrixA, matrixSize, matrixSize);
	matrixInit.run(matrixB, matrixSize, matrixSize);
	matrixInit.wait();

	/*
	 * do computation
	 */
	double runtime[4];
	Task<MatrixMulSGPU<FTYPE>> mm(ProcList(0), TaskType::gpu_task);
	mm.init(matrixA, matrixB, matrixC, matrixSize, matrixSize, matrixSize);
	mm.run(runtime, blockSize, memtype);
	mm.wait();


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

	delete matrixA;
	delete matrixB;
	delete matrixC;
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tMatrix size: "<<matrixSize<<" X "<<matrixSize<<std::endl;
		std::cout<<"\tcuda Block size: "<<blockSize<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		//std::cout<<"\t\tmemory create time: "<<memcreateTime<<" s"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<std::fixed<<std::setprecision(4)<<runtime[3]<<"(s)"<<std::endl;

	}
	return 0;

}







