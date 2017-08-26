#include "Utc.h"
#include "../helper_getopt.h"

#include <iostream>

using namespace iUtc;
class BlockMatrixMultiply: public UserTaskBase{
private:
	int dimRows;
	int dimColums;
	int blockRows;
	int blockColums;
	int nBlocks;

	int subMatrixColumIdx;
	double *subMatrixA;
	double *tmpMatrixA;
	double *subMatrixB;
	double *subMatrixC;

	Conduit *leftNeighbour;
	Conduit *rightNeighbour;

public:
	void initImpl(int dimRows, int dimColums, int nBlocks, int columIdx,
			Conduit* left, Conduit* right){
		if(__numProcesses !=1){
			std::cerr<<"This task only run on single node !!!\n";
			exit(1);
		}
		if(__localThreadId==0){
			this->dimRows = dimRows;
			this->dimColums = dimColums;
			this->nBlocks = nBlocks;
			blockColums = dimColums / nBlocks;
			blockRows = dimRows / __numLocalThreads;
			subMatrixColumIdx = columIdx;
			leftNeighbour = left;
			rightNeighbour = right;
			subMatrixA = (double*)malloc(dimRows * blockColums * sizeof(double));
			subMatrixB = (double*)malloc(dimRows * blockColums * sizeof(double));
			subMatrixC = (double*)malloc(dimRows * blockColums * sizeof(double));
		}

		inter_Barrier();

		/* init local submatrx with all threads */
		for(int i = __localThreadId * blockRows; i< (__localThreadId+1)*blockRows; i++){
			for( int j =0; j< blockColums; j++ ){
				subMatrixA[i*blockColums + j] = i+ 1*j + 1*__processId +1;
				subMatrixB[i*blockColums + j] = i+ 2*j + 2*__processId +1;
				subMatrixC[i*blockColums + j] = 0;
			}
		}
		intra_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
		}
	}

	void runImpl(double *runtime){
		intra_Barrier();
		Timer timer;
		timer.start();
		/* compute local partial C[][] with local A[][] and B[][] */
		int startRowA = __localThreadId * blockRows;
		int currentColumIdx = subMatrixColumIdx;
		int startRowB = currentColumIdx * blockColums;
		//std::cout<<startRowA<<blockRows<<blockColums<<std::endl;
		for(int i= startRowA; i< startRowA + blockRows; i++){
			for(int k = 0; k< blockColums; k++){
			for(int j = 0; j< blockColums; j++){
				//for(int k = 0; k< blockColums; k++){
					subMatrixC[i*blockColums + j] += subMatrixA[i*blockColums + k] *
														subMatrixB[(startRowB + k)*blockColums + j];
				}
			}
		}

		/* rotate matrixA to compute */
		if(nBlocks>1 && subMatrixColumIdx == nBlocks-1&&getUniqueExecution()){
			tmpMatrixA = (double*)malloc(dimRows * blockColums * sizeof(double));
		}
		intra_Barrier();

		for( int i=1; i< nBlocks; i++){

			/* the right end task do read---write, other task do write---read
			 * in case of causing dead lock with a ring op
			 */
			if(subMatrixColumIdx == nBlocks-1){
				if(getUniqueExecution){
					memcpy(tmpMatrixA, subMatrixA, dimRows * blockColums * sizeof(double));
				}
				//std::cout<<__localThreadId<<" "<<__processId<<" "<<ERROR_LINE<<std::endl;
				intra_Barrier();
				rightNeighbour->Read(subMatrixA, sizeof(double)*dimRows*blockColums, i);
				leftNeighbour->Write(tmpMatrixA, sizeof(double)*dimRows*blockColums, i);

			}else{
				/* send local matrix A to left neighbour task */
				leftNeighbour->Write(subMatrixA, sizeof(double)*dimRows*blockColums, i);
				/* get remote matrix A from right neighbour task */
				rightNeighbour->Read(subMatrixA, sizeof(double)*dimRows*blockColums, i);
			}
			//std::cout<<__localThreadId<<" "<<__processId<<" "<<ERROR_LINE<<std::endl;
			intra_Barrier();
			currentColumIdx = (subMatrixColumIdx + i) % nBlocks;
			startRowB = currentColumIdx * blockColums;
			for(int i= startRowA; i< startRowA + blockRows; i++){
				for(int j = 0; j< blockColums; j++){
					for(int k = 0; k< blockColums; k++){
						subMatrixC[i*blockColums + j] += subMatrixA[i*blockColums + k] *
															subMatrixB[(startRowB + k)*blockColums + j];
					}
				}
			}
		}

		intra_Barrier();
		runtime[__localThreadId]=timer.stop();
		if(tmpMatrixA && __localThreadId == 0)
			free(tmpMatrixA);
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
			//std::cout<<subMatrixC[1*blockColums+ 10 ]<<std::endl;
			//std::cout<<subMatrixC[10*blockColums+100]<<std::endl;
		}
	}

	~BlockMatrixMultiply(){
		if(subMatrixA)
			free(subMatrixA);
		if(subMatrixB)
			free(subMatrixB);
		if(subMatrixC)
			free(subMatrixC);
	}
};

#define TASK_4
int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads;
	int nprocs;
	int matrixSize;

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
	int myproc = ctx.getProcRank();

#ifdef TASK_1
	double *runtime = new double[nthreads];
	Task<BlockMatrixMultiply> subMM1(ProcList(nthreads, 0));
	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = 1;

	subMM1.init(dimRows, dimColums, nBlocks, 0, nullptr, nullptr);
	subMM1.run(runtime);
	subMM1.wait();

	double avg_runtime1=0;
	for(int i =0; i<nthreads; i++)
		avg_runtime1+= runtime[i];
	avg_runtime1/=nthreads;
	std::cout<<"average run() time: "<<avg_runtime1<<std::endl;
	delete runtime;
#endif

#ifdef TASK_2
	double *runtime1 = new double[nthreads];
	double *runtime2 = new double[nthreads];
	Task<BlockMatrixMultiply> subMM1(ProcList(nthreads, 0));
	Task<BlockMatrixMultiply> subMM2(ProcList(nthreads, 1));
	Conduit cdt12(&subMM1, &subMM2);

	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = 2;

	subMM1.init(dimRows, dimColums, nBlocks, 0, &cdt12, &cdt12);
	subMM2.init(dimRows, dimColums, nBlocks, 1, &cdt12, &cdt12);

	subMM1.run(runtime1);
	subMM2.run(runtime2);

	subMM1.wait();
	subMM2.wait();

	double avg_runtime1=0;
	double avg_runtime2=0;
	for(int i =0; i<nthreads; i++){
		avg_runtime1+= runtime1[i];
		avg_runtime2+= runtime2[i];
	}
	avg_runtime1/=nthreads;
	avg_runtime2/=nthreads;
	std::cout<<"average run() time: "<<avg_runtime1<<" "<<avg_runtime2<<std::endl;
	delete runtime1;
	delete runtime2;
#endif

#ifdef TASK_4
	double *runtime1 = new double[nthreads];
	double *runtime2 = new double[nthreads];
	double *runtime3 = new double[nthreads];
	double *runtime4 = new double[nthreads];
	Task<BlockMatrixMultiply> subMM1(ProcList(nthreads, 0));
	Task<BlockMatrixMultiply> subMM2(ProcList(nthreads, 1));
	Task<BlockMatrixMultiply> subMM3(ProcList(nthreads, 2));
	Task<BlockMatrixMultiply> subMM4(ProcList(nthreads, 3));
	Conduit cdt12(&subMM1, &subMM2);
	Conduit cdt23(&subMM2, &subMM3);
	Conduit cdt34(&subMM3, &subMM4);
	Conduit cdt41(&subMM4, &subMM1);
	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = 4;

	subMM1.init(dimRows, dimColums, nBlocks, 0, &cdt41, &cdt12);
	subMM2.init(dimRows, dimColums, nBlocks, 1, &cdt12, &cdt23);
	subMM3.init(dimRows, dimColums, nBlocks, 2, &cdt23, &cdt34);
	subMM4.init(dimRows, dimColums, nBlocks, 3, &cdt34, &cdt41);

	subMM1.run(runtime1);
	subMM2.run(runtime2);
	subMM3.run(runtime3);
	subMM4.run(runtime4);

	subMM1.wait();
	subMM2.wait();
	subMM3.wait();
	subMM4.wait();

	ctx.Barrier();

	double avg_runtime1=0;
	double avg_runtime2=0;
	double avg_runtime3=0;
	double avg_runtime4=0;
	for(int i =0; i<nthreads; i++){
		avg_runtime1+= runtime1[i];
		avg_runtime2+= runtime2[i];
		avg_runtime3+= runtime3[i];
		avg_runtime4+= runtime4[i];
	}
	avg_runtime1/=nthreads;
	avg_runtime2/=nthreads;
	if(myproc ==0)
		std::cout<<"average run() time: "<<std::endl;
	if(myproc ==0)
		std::cout<<avg_runtime1<<std::endl;
	ctx.Barrier();
	if(myproc == 1)
		std::cout<<avg_runtime2<<std::endl;
	ctx.Barrier();
	if(myproc == 2)
		std::cout<<avg_runtime3<<std::endl;
	ctx.Barrier();
	if(myproc ==3)
		std::cout<<avg_runtime4<<std::endl;
	delete runtime1;
	delete runtime2;
	delete runtime3;
	delete runtime4;
#endif

	//delete &ctx;
	return 0;

}
