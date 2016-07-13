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

	int subMatrixColumIdx;
	double *subMatrixA;
	double *subMatrixB;
	double *subMatrixC;

	Conduit *leftNeighbour;
	Conduit *rightNeighbour;

public:
	void initImpl(int dimRows, int dimColums, int nBlocks, int columIdx, double randseed,
			Conduit* left, Conduit* right){
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
				subMatrixA[i*blockColums + j] = i*blockColums + j + randseed;
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
		int currentColumIdx = subMatrixColumIdx;
		int startRowB = currentColumIdx * blockColums;
		for(int i= startRowA; i< startRowA + blockRows; i++){
			for(int j = 0; j< blockColums; j++){
				for(int k = 0; k< blockColums; k++){
					subMatrixC[i*blockColums + j] += subMatrixA[i*blockColums + k] *
														subMatrixB[(startRowB + k)*blockColums + j];
				}
			}
		}

		/* rotate matrixA to compute */
		double *tmpMatrixA = nullptr;
		if(subMatrixColumIdx == nBlocks-1&&getUniqueExecution()){
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
				intra_Barrier();
				rightNeighbour->Read(subMatrixA, sizeof(double)*dimRows*blockColums, i);
				leftNeighbour->Write(tmpMatrixA, sizeof(double)*dimRows*blockColums, i);

			}else{
				/* send local matrix A to left neighbour task */
				leftNeighbour->Write(subMatrixA, sizeof(double)*dimRows*blockColums, i);
				/* get remote matrix A from right neighbour task */
				rightNeighbour->Read(subMatrixA, sizeof(double)*dimRows*blockColums, i);
			}
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
		if(tmpMatrixA)
			free(tmpMatrixA);
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<"finish runImpl.\n";
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
