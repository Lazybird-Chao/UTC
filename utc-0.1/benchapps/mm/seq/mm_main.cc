/*
 * mm_main.cc
 *
 *
 */

#include <iostream>
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"

#define FTYPE float


int main(int argc, char **argv){
	int matrixSize = 1024;
	bool printTime = false;

	/*
	 * run as ./a.out -v -s 100
	 * 		-v: print time info
	 * 		-s: the size of matrix, we assume a square matrix
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vs:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vs:");
	}
	if(matrixSize<=0){
		std::cerr<<"matrix size is illegal"<<std::endl;
		exit(1);
	}

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

	/*
	 * main computing
	 */
	double t1, t2;
	t1 = getTime();
	for(int i=0; i<matrixSize; i++){
		for(int j=0; j<matrixSize; j++){
			FTYPE tmp = 0;
			for(int k=0; k<matrixSize; k++)
				tmp +=matrixA[i*matrixSize +k] * matrixB[k*matrixSize +j];
			matrixC[i*matrixSize + j] = tmp;
		}
	}
	t2 = getTime();
	double runtime = t2-t1;

	free(matrixA);
	free(matrixB);
	free(matrixC);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tMatrix info: "<<matrixSize<<" X "<<matrixSize<<std::endl;
		std::cout<<"\tTime info: "<<runtime<<"(s)"<<std::endl;
	}

}



