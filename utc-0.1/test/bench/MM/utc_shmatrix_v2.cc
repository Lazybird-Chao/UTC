#include "Utc.h"
#include "../helper_getopt.h"
#include "../helper_printtime.h"

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
	double *localPtrA[2];
	double *subMatrixB;
	double *subMatrixC;


public:
	void initImpl(int dimRows, int dimColums, int nBlocks){
		if(__numProcesses>1 && __numLocalThreads <2){
			std::cerr<<"Error, this task requires at last 2 local threads when run with "
					"multiple processes.\n";
			exit(1);
		}
		if(__localThreadId==0){
			this->dimRows = dimRows;
			this->dimColums = dimColums;
			this->nBlocks = nBlocks;
			blockColums = dimColums / nBlocks;

			subMatrixA = new GlobalScopedData<double>(this, dimRows * blockColums);
			subMatrixB = (double*)malloc(dimRows * blockColums * sizeof(double));
			subMatrixC = (double*)malloc(dimRows * blockColums * sizeof(double));
			if(__numProcesses>1){
				localPtrA[0] = (double*)malloc(dimRows * blockColums * sizeof(double));
				localPtrA[1] = (double*)malloc(dimRows * blockColums * sizeof(double));
			}

		}
		intra_Barrier();
		/* init local submatrx with all threads */
		for(int i = __localThreadId * (dimRows/__numLocalThreads); i< (__localThreadId+1)*(dimRows/__numLocalThreads); i++){
			for( int j =0; j< blockColums; j++ ){
				subMatrixA->store(i+ 1*j + 1*__processId +1, i*blockColums+j);
				subMatrixB[i*blockColums + j] = i+ 2*j + 2*__processId +1;
				subMatrixC[i*blockColums + j] = 0;
			}
		}
		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
		}
	}

	void runImpl(double *runtime){
		Timer timer, timer1;

		/* compute local partial C[][] with local A[][] and B[][] */
		int NLINES = 50;
		if(__numProcesses>1)
			blockRows = (dimRows-NLINES) / (__numLocalThreads-1);
		else
			blockRows = dimRows / __numLocalThreads;

		int startRowA = __localThreadId * blockRows;
		int endRowA = startRowA + blockRows;
		if(__localThreadId == __numLocalThreads -2)
			endRowA = dimRows - NLINES;
		int currentColumIdx;
		int startRowB;
		double *currentA = subMatrixA->getPtr();
		int neighbour= __processId;
		int idxA=0;
		inter_Barrier();
		timer.start();
		for(int iterBlock = 0; iterBlock <nBlocks; iterBlock++){
			timer1.start();
			currentColumIdx = (__processId + iterBlock) % nBlocks;
			startRowB = currentColumIdx * blockColums;
			if((__numProcesses>1 &&__localThreadId < __numLocalThreads-1)
					|| (__numProcesses==1)){
				if(iterBlock >0){
					currentA = localPtrA[idxA];
					idxA = (idxA+1)%2;
				}
				// thread 0 to n-1 do computing
				for(int i= startRowA; i< endRowA; i++){
					for(int k = 0; k< blockColums; k++){
					for(int j = 0; j< blockColums; j++){
						//for(int k = 0; k< blockColums; k++){
							subMatrixC[i*blockColums + j] += currentA[i*blockColums + k] *
																subMatrixB[(startRowB + k)*blockColums + j];
						}
					}
				}
			}
			else{
				if(iterBlock > 0){
					currentA = localPtrA[(idxA+1)%2];
				}
				for(int i= dimRows-NLINES; i< dimRows; i++){
					for(int k = 0; k< blockColums; k++){
					for(int j = 0; j< blockColums; j++){
						//for(int k = 0; k< blockColums; k++){
							subMatrixC[i*blockColums + j] += currentA[i*blockColums + k] *
																subMatrixB[(startRowB + k)*blockColums + j];
						}
					}
				}
				// the last thread prefetch data
				if(__numProcesses > 1 && iterBlock<nBlocks-1){
					neighbour = (neighbour+1)%__numProcesses;
					subMatrixA->rloadblock(neighbour, localPtrA[idxA], 0, dimRows * blockColums);
				}
				idxA = (idxA+1)%2;
			}
			runtime[__localThreadId*2+1] += timer1.stop();
			inter_Barrier();
		}
		//std::cout<<ERROR_LINE<<std::endl;
		//inter_Barrier();
		runtime[__localThreadId*2] = timer.stop();
		if(__numProcesses>1 && getUniqueExecution()){
			free(localPtrA[0]);
			free(localPtrA[1]);
		}
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
	double *runtime = new double[2*nthreads];
	for(int i=0; i<2*nthreads;i++)
		runtime[i]=0;

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
		avg_runtime1+= runtime[i*2];
	double first_runtime=0;
	for(int i=0; i<nthreads-1; i++)
		first_runtime += runtime[i*2+1];
	first_runtime /= (nthreads-1);
	avg_runtime1/=nthreads;
	MPI_Reduce(&avg_runtime1, &avg_runtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avg_runtime2/=nprocs;
	if(myproc==0){
		std::cout<<"average run() time: "<<avg_runtime1<<std::endl;
		std::cout<<"last thread run() time: "<<runtime[(nthreads-1)*2+1]<<std::endl;
		std::cout<<"first thread run() time: "<<first_runtime<<std::endl;

		double timer[3];
		timer[0] = avg_runtime2;
		timer[1] = runtime[(nthreads-1)*2+1];
		timer[2] = first_runtime;
		print_time(3, timer);
	}
	delete runtime;
	//std::cout<<ERROR_LINE<<std::endl;
	return 0;

}
