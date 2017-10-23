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
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#define FTYPE float



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
		case 'o':
			outfile = optarg;
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
	char *infileA = nullptr;
	char *infileB = nullptr;
	infileA = "../input/1536_8k_A.txt";
	infileB = "../input/8k_1536_B.txt";

	FTYPE *matrixA;
	FTYPE *matrixB;
	FTYPE *matrixC;

	int sizeM, sizeN, sizeP;
	sizeM = 1536*5;
	sizeN = 8192;
	sizeP = 1536*5;

	if(infileA == nullptr){
		matrixA = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeN);
		matrixB = (FTYPE*)malloc(sizeof(FTYPE)*sizeN*sizeP);
		matrixC = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeP);

		FTYPE rnumber = (FTYPE)(rand()%100)/(rand()%10);
		for(int i=0; i<sizeM; i++){
			for(int j=0; j<sizeN; j++){
				matrixA[i*sizeN + j] = (j + rnumber)/matrixSize + i;
				matrixB[i*sizeN + j] = (j - rnumber)/matrixSize + i;
			}
		}
	} else{
		fromFile(matrixA, sizeM, sizeN, infileA, true);
		fromFile(matrixB, sizeN, sizeP, infileB, true);
		matrixC = (FTYPE*)malloc(sizeof(FTYPE)*sizeM*sizeP);
	}

	toFile(matrixA, sizeM, sizeN, "1536_8k_A.txt", true);
	toFile(matrixB, sizeN, sizeP, "8k_1536_B.txt", true);
	return 0;


	/*
	 * main computing
	 */
	double t1, t2;
	t1 = getTime();
	for(int i=0; i<sizeM; i++){
		for(int j=0; j<sizeP; j++){
			FTYPE tmp = 0;
			for(int k=0; k<sizeN; k++)
				tmp +=matrixA[i*sizeN +k] * matrixB[j*sizeN +k];
			matrixC[i*sizeM + j] = tmp;
		}
	}
	t2 = getTime();
	double runtime = t2-t1;


	free(matrixA);
	free(matrixB);
	free(matrixC);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tMatrix info: "<<sizeM<<" X "<<sizeN<<" X "<<sizeP<<std::endl;
		std::cout<<"\tTime info: "<<std::fixed<<std::setprecision(4)<<runtime<<"(s)"<<std::endl;
	}

	runtime *=1000;
	print_time(1, &runtime);

	return 0;

}



