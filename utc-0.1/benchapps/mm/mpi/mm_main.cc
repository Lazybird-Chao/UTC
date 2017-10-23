/*
 * mm_main.cc
 *
 * The sequential matrix multiply program
 *
 * usage:
 * 		compile with Makefile
 * 		run as: ./a.out -v -s 100
 * 				-v: print time info
 * 				-s: the size of matrix , we assume a square matrix
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include "mpi.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#define FTYPE float
#define MPI_FTYPE MPI_FLOAT



void fromFile(FTYPE* &matrix, int& h, int& w, const char* file, bool isBinary){
		std::ifstream infile;
		if(isBinary)
			infile.open(file, std::ios::binary);
		else
			infile.open(file);
		if(isBinary){
			infile.read((char*)&h, sizeof(int));
			infile.read((char*)&w, sizeof(int));
			matrix = new FTYPE[w*h];
			infile.read((char*)matrix, h*w*sizeof(FTYPE));
		}
		else{
			infile>>h;
			infile>>w;
			matrix = new FTYPE[w*h];
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++)
					infile>>matrix[i*w+j];
			}
		}
}

void toFile(FTYPE* matrix, int h, int w, const char* file, bool isBinary){
		std::ofstream outfile;
		if(isBinary)
			outfile.open(file, std::ios::binary);
		else
			outfile.open(file);
		if(isBinary){
			outfile.write((char*)&h, sizeof(int));
			outfile.write((char*)&w, sizeof(int));
			outfile.write((char*)matrix, h*w*sizeof(FTYPE));
		}
		else{
			outfile<<h<<" "<<w<<std::endl;
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++)
					outfile<<matrix[i*w+j]<<" ";
				outfile<<std::endl;
			}
		}
}

int main(int argc, char **argv){
	int matrixSize = 1024;
	bool printTime = false;
	char *outfile = nullptr;
	int nprocess = 1;

	/*
	 * run as ./a.out -v -s 100
	 * 		-v: print time info
	 * 		-s: the size of matrix, we assume a square matrix
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vs:p:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 's':
			matrixSize = atoi(optarg);
			break;
		case 'o':
			outfile = optarg;
			break;
		case 'p':
			nprocess = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vs:p:");
	}
	if(matrixSize<=0){
		std::cerr<<"matrix size is illegal"<<std::endl;
		exit(1);
	}

	/*
	 * init mpi environment
	 */
	MPI_Init(&argc, &argv);
	int procs;;
	int myproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

//	char nodename[1024];
//	int nodenamelen;
//	MPI_Get_processor_name(nodename, &nodenamelen);
//	std::cout<<nodename<<": "<<myproc<<": "<<procs<<std::endl;

	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}

	/*
	 * create matrix and initialize with random number
	 */
	//char *infileA = nullptr;
	//char *infileB = nullptr;
	char* infileA = "../input/1536_8k_A.txt";
	char* infileB = "../input/8k_1536_B.txt";

	FTYPE *matrixA = nullptr;
	FTYPE *matrixB = nullptr;
	FTYPE *matrixC = nullptr;

	FTYPE *localBlockA;
	FTYPE *localBlockB;
	FTYPE *localBlockC;
	int sizeM, sizeN, sizeP;

	if(myproc == 0){
		if(infileA == nullptr){
			sizeM = sizeN = sizeP = matrixSize;
			matrixA = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeN);
			matrixB = (FTYPE*)malloc(sizeof(FTYPE)*sizeN*sizeP);
			matrixC = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeP);

			FTYPE rnumber = (FTYPE)(rand()%100)/(rand()%10);
			for(int i=0; i<sizeM; i++){
				for(int j=0; j<sizeP; j++){
					matrixA[i*sizeN + j] = (j + rnumber)/matrixSize + i;
					matrixB[i*sizeN + j] = (j - rnumber)/matrixSize + i;
				}
			}
		} else{
			fromFile(matrixA, sizeM, sizeN, infileA, true);
			fromFile(matrixB, sizeN, sizeP, infileB, true);
			matrixC = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeP);
		}

		//toFile(matrixA, matrixSize, matrixSize, "16k_16k_A.txt", true);
		//toFile(matrixB, matrixSize, matrixSize, "16k_16k_B.txt", true);
		//return 0;
	}
	MPI_Bcast(&sizeM, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sizeN, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sizeP, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int blockRows = sizeM / nprocess;
	localBlockA = new FTYPE[blockRows*sizeN];
	localBlockB = new FTYPE[blockRows*sizeN];
	localBlockC = new FTYPE[blockRows*sizeN];
	if(matrixA == nullptr){
		matrixA = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeN);
	}
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;
	double t0;
	double t1;
	MPI_Barrier(MPI_COMM_WORLD);
	if(myproc == 0)
		std::cout<<"start computing...\n"<<sizeM<<" "<<sizeN<<" "<<sizeP<<std::endl;
	/*
	 * scatter matrixB to all processes
	 */
	t0 = MPI_Wtime();
	t1 = t0;
	MPI_Scatter(matrixB, blockRows*sizeN, MPI_FTYPE,
				localBlockB, blockRows*sizeN, MPI_FTYPE,
				0, MPI_COMM_WORLD);

	MPI_Bcast(matrixA, sizeM*sizeN, MPI_FTYPE, 0, MPI_COMM_WORLD);

	commtime += MPI_Wtime() - t1;

	/*
	 * main computing
	 */
	for(int p = 0; p < nprocess; p++){
		/*
		 * bcast block of matrixA
		 */
		/*
		if(myproc == 0)
			memcpy(localBlockA, matrixA + p*blockRows*matrixSize, blockRows*matrixSize*sizeof(FTYPE));
		t1 = MPI_Wtime();
		MPI_Bcast(localBlockA, blockRows*matrixSize, MPI_FTYPE, 0, MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t1;
		*/
		memcpy(localBlockA, matrixA + p*blockRows*sizeN, blockRows*sizeN*sizeof(FTYPE));
		/*
		 * local compute
		 */
		t1 = MPI_Wtime();
		FTYPE *c_start = localBlockC + p*blockRows*blockRows;
		for(int i=0; i<blockRows; i++){
			for(int j=0; j<blockRows; j++){
				FTYPE tmp = 0;
				for(int k=0; k<sizeN; k++)
					tmp +=localBlockA[i*sizeN +k] * localBlockB[j*sizeN +k];
				c_start[i*blockRows + j] = tmp;
			}
		}
		computetime += MPI_Wtime() - t1;
	}
	/*
	 * gather block of matrixC
	 */
	t1 = MPI_Wtime();
	MPI_Gather(localBlockC, blockRows*sizeM, MPI_FTYPE,
			   matrixC, blockRows*sizeM, MPI_FTYPE,
			   0, MPI_COMM_WORLD);
	commtime += MPI_Wtime() - t1;
	MPI_Barrier(MPI_COMM_WORLD);
	totaltime = MPI_Wtime() - t0;

	if(myproc == 0){
		//delete(matrixA);
		//delete(matrixB);
		free(matrixC);
	}
	delete(localBlockA);
	delete(localBlockB);
	delete(localBlockC);

	double runtime[3]={0,0,0};
	MPI_Reduce(&totaltime, runtime+0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&computetime, runtime+1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&commtime, runtime+2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(myproc == 0){
		for(int i = 0; i< 3; i++)
			runtime[i] /= nprocess;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tMatrix info: "<<sizeM<<" X "<<sizeN<<" X "<<sizeP<<std::endl;
			std::cout<<"\tTime info: \n";
			std::cout<<"\t\ttotaltime: "<<std::fixed<<std::setprecision(4)<<runtime[0]<<"(s)"<<std::endl;
			std::cout<<"\t\tcomputetime: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
			std::cout<<"\t\tcommtime: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		}

		for(int i = 0; i< 3; i++)
			runtime[i] *=1000;
		print_time(3, runtime);
	}

	MPI_Finalize();
	return 0;

}



