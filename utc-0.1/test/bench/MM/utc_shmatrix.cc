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

	GlobalScopedData<double> *subMatrixA;
	double *localPtrA;
	double *subMatrixB;
	double *subMatrixC;


public:
	void initImpl(int dimRows, int dimColums, int nBlocks){

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
				subMatrixA->store(i+ 1*j + 1*__processId +1, i*blockColums+j);
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
		Timer timer;
		intra_Barrier();
		timer.start();
		/* compute local partial C[][] with local A[][] and B[][] */
		int startRowA = __localThreadId * blockRows;
		int currentColumIdx = __processId;
		int startRowB = currentColumIdx * blockColums;
		for(int i= startRowA; i< startRowA + blockRows; i++){
			for(int k = 0; k< blockColums; k++){
			for(int j = 0; j< blockColums; j++){
				//for(int k = 0; k< blockColums; k++){
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
		runtime[__localThreadId] = timer.stop();
		if(__numProcesses>1 && getUniqueExecution())
			free(localPtrA);
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
		}
	}

	~BlockMatrixMultiply(){
		if(subMatrixA)
			delete subMatrixA;
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
		opt=getopt(argc, argv, "t:p:s:");
	}
	int procs = ctx.numProcs();
	if(nprocs != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	int myproc = ctx.getProcRank();

	double *runtime = new double[nthreads];
	ProcList plist1;
	for(int i=0; i<nprocs; i++)
		for(int j=0; j<nthreads; j++)
			plist1.push_back(i);
	Task<BlockMatrixMultiply> subMM1(plist1);
	int dimRows = matrixSize;
	int dimColums = matrixSize;
	int nBlocks = nprocs;
	subMM1.init(dimRows, dimColums, nBlocks);
	subMM1.run(runtime);
	subMM1.wait();

	double avg_runtime1=0;
	double avg_runtime2=0;
	for(int i =0; i<nthreads; i++)
		avg_runtime1+= runtime[i];
	avg_runtime1/=nthreads;
	MPI_Reduce(&avg_runtime1, &avg_runtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avg_runtime2/=nprocs;
	if(myproc==0)
		std::cout<<"average run() time: "<<avg_runtime2<<std::endl;
	delete runtime;
	//std::cout<<ERROR_LINE<<std::endl;
	return 0;

}
