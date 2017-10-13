/*
 * mm_main.cc
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"

#include "task.h"

#define FTYPE float
#define MAX_THREADS 64

int main(int argc, char** argv){
	bool printTime = false;
	int matrixSize = 1024;
	char *fileout = nullptr;
	char *filein = nullptr;

	int nthreads=1;
	int nprocess=1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	if(ctx.getProcRank() == 0)
		std::cout<<"UTC context initialized !\n";

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vt:p:s:o:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case 'o':
			fileout = optarg;
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vt:p:s:o:");
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

	/*
	 * create matrix and initialize with random number
	 */
	char *infileA = nullptr;
	char *infileB = nullptr;
	infileA = "../input/16k_16k_A.txt";
	infileB = "../input/16k_16k_B.txt";
	FTYPE *matrixA = nullptr;
	FTYPE *matrixB = nullptr;
	Task<RandomMatrixGen<FTYPE>> matrixInit(ProcList(0));
	matrixInit.run(&matrixA, &matrixSize, &matrixSize, infileA, true);
	matrixInit.wait();
	matrixInit.run(&matrixB, &matrixSize, &matrixSize, infileB, true);
	matrixInit.wait();

	//toFile(matrixA, matrixSize, matrixSize, "4k_4k_A.txt", true);
	//toFile(matrixB, matrixSize, matrixSize, "4k_4k_B.txt", true);

	FTYPE *matrixC = nullptr;
	if(ctx.getProcRank() == 0)
		matrixC = new FTYPE[matrixSize*matrixSize];

	/*
	 * do computation
	 */
	double runtime_m[MAX_THREADS][3];
	ProcList plist;
	for(int i = 0; i < procs; i++)
		for(int j = 0; j<nthreads; j++)
			plist.push_back(i);
	Task<MatrixMulWorker<FTYPE>> mm(plist, TaskType::cpu_task);
	mm.init(matrixA, matrixB, matrixC, matrixSize, matrixSize, matrixSize);
	mm.run(runtime_m);
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
	mm.finish();
	if(ctx.getProcRank() == 0){
		if(fileout){
			toFile(matrixC, matrixSize, matrixSize, fileout, true);
		}
		delete matrixA;
		delete matrixB;
		delete matrixC;

		double runtime[3]={0,0,0};
		for(int i=0; i<nthreads; i++)
			for(int j=0; j<3; j++)
				runtime[j]+= runtime_m[i][j];
		for(int j=0; j<3; j++)
			runtime[j] /= nthreads;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tMatrix size: "<<matrixSize<<" X "<<matrixSize<<std::endl;
			std::cout<<"\tTime info: "<<std::endl;
			std::cout<<"\t\ttotal run time: "<<std::fixed<<std::setprecision(4)<<runtime[0]<<"(s)"<<std::endl;
			std::cout<<"\t\tcompute time: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
			std::cout<<"\t\tcomm time: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;

		}
		//std::cout<<ERROR_LINE<<std::endl;
		for(int i=0; i<3; i++)
			runtime[i] *= 1000;
		print_time(3, runtime);
	}
	ctx.Barrier();
	return 0;
}
