#include "Utc.h"
#include "../bench/helper_getopt.h"

#include <iostream>

using namespace iUtc;
class BlockMatrixMultiply: public UserTaskBase{
private:
	int dimRows;
	int dimColums;
	int blockRows;
	int blockColums;
	int nBlocks;

	GlobalScopedData<double> *subMatrixA;
	double *localPtrA;
	double *subMatrixB;
	double *subMatrixC;


public:
	void initImpl(int dimRows, int dimColums, int nBlocks, double randseed,
			){
		if(__numProcesses !=1){
			std::cout<<"This task only run on single node !!!\n";
			exit(1);
		}
		if(__localThreadId==0){
			this->dimRows = dimRows;
			this->dimColums = dimColums;
			this->nBlocks = nBlocks;
			blockColums = dimColums / nBlocks;
			blockRows = dimRows / __numLocalThreads;

			subMatrixA = new GlobalScopedData<double>(this, dimRows * blockColums);
			localPtrA = subMatrixA->getPtr();
			subMatrixB = (double*)malloc(dimRows * blockColums * sizeof(double));
			subMatrixC = (double*)malloc(dimRows * blockColums * sizeof(double));
		}
		inter_Barrier();
		/* init local submatrx with all threads */
		for(int i = __localThreadId * blockRows; i< (__localThreadId+1)*blockRows; i++){
			for( int j =0; j< blockColums; j++ ){
				subMatrixA->store(i*blockColums + j + randseed, i*blockColums+j);
				subMatrixB[i*blockColums + j] = i*blockColums + j - randseed;
				subMatrixC[i*blockColums + j] = 0;
			}
		}
		intra_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<"finish initImpl.\n";
		}
	}

	void runImpl(){
		/* compute local partial C[][] with local A[][] and B[][] */
		int startRowA = __localThreadId * blockRows;
		int currentColumIdx = __processId;
		int startRowB = currentColumIdx * blockColums;
		for(int i= startRowA; i< startRowA + blockRows; i++){
			for(int j = 0; j< blockColums; j++){
				for(int k = 0; k< blockColums; k++){
					subMatrixC[i*blockColums + j] += localPtrA[i*blockColums + k] *
														subMatrixB[(startRowB + k)*blockColums + j];
				}
			}
		}

		/* get remote matrixA to compute */
		if(__numProcesses>1 && getUniqueExecution()){
			localPtrA= (double*)malloc(dimRows * blockColums * sizeof(double));
		}
		intra_Barrier();
		int neighbour= __processId;
		for( int i=1; i< nBlocks; i++){
			neighbour = (neighbour+1)%__numProcesses;
			if(getUniqueExecution()){
				subMatrixA->rloadblock(neighbour, localPtrA, 0, dimRows * blockColums);
			}
			intra_Barrier();

			currentColumIdx = (__processId + i) % nBlocks;
			startRowB = currentColumIdx * blockColums;
			for(int i= startRowA; i< startRowA + blockRows; i++){
				for(int j = 0; j< blockColums; j++){
					for(int k = 0; k< blockColums; k++){
						subMatrixC[i*blockColums + j] += localPtrA[i*blockColums + k] *
															subMatrixB[(startRowB + k)*blockColums + j];
					}
				}
			}
		}

		intra_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<"finish runImpl.\n";
		}
	}
};

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
		opt=getopt(argc, argv, "t:p:");
	}
	int nproc = ctx.numProcs();
	int myproc = ctx.getProcRank();

	Task<BlockMatrixMultiply> subMM1(ProcList(nthreads, 0));
	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = 1;
	subMM1.init(dimRows, dimColums, nBlocks, 0, 0.3, nullptr, nullptr);
	subMM1.run();
	subMM1.wait();


	/*Task<BlockMatrixMultiply> subMM1(ProcList(nthreads, 0));
	Task<BlockMatrixMultiply> subMM2(ProcList(nthreads, 1));
	Conduit cdt12(&subMM1, &subMM2);

	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = 2;

	subMM1.init(dimRows, dimColums, nBlocks, 0, 0.3, &cdt12, &cdt12);
	subMM2.init(dimRows, dimColums, nBlocks, 1, 0.5, &cdt12, &cdt12);

	subMM1.run();
	subMM2.run();

	subMM1.wait();
	subMM2.wait();*/

	return 0;

}
