/*
 * hc_main.cc
 *
 * The single GPU heat conduction program. Single cuda stream.
 * In this program we have two cuda kernels, one do jacabi compute, the other check
 * the convergence.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -h 100 -w 80 -e 0.001 -b 16
 * 			-v: print time info
 * 			-h: 2D domain height
 * 			-w: 2D domain width
 * 			-e: convergence accuracy
 * 			-b: cuda block size.[16, 16, 1]
 */

#include <iostream>
#include <math.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_err.h"
#include "hc_kernel.h"

#define FTYPE float

#define H 1.0
#define T_SRC0 550.0
#define ITERMAX 100		// not used

void init_domain(float *domain_ptr, int h, int w){
	for (int j = 0; j < (int)floor(h/H); j++) {
		for (int i = 0; i < (int) floor (w / H); i++) {
			domain_ptr[j*((int) floor (w / H)) + i] = 0.0;
		}
	}
}

FTYPE get_convergence_sqd(FTYPE *sqd_array, int w){
	FTYPE sum = 0.0;
	for(int i=0; i< w; i++)
		sum += sqd_array[i];
	return sum;
}



int main(int argc, char**argv){
	int WIDTH = 20;
	int HEIGHT = 20;
	FTYPE EPSILON = 0.1;
	bool printTime = false;
	int blockSize = 16;

	/*
	 * run as ./a.out -v -h 100 -w 80 -e 0.001
	 * 		-v: print time info
	 * 		-h: 2D domain height
	 * 		-w: 2D domain width
	 * 		-e: convergence accuracy
	 * 		-b: cuda block size
	 */
	int opt;
	extern char* optarg;
	extern int optind, optopt;
	opt=getopt(argc, argv, "vh:w:e:b:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 'w':
			WIDTH = atoi(optarg);
			break;
		case 'h':
			HEIGHT = atoi(optarg);
			break;
		case 'e':
			EPSILON = atof(optarg);
			break;
		case 'b':
			blockSize = atoi(optarg);
			break;
		case ':':
			std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
			break;
		case '?':
			std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vh:w:e:");
	}
	if(WIDTH<=0 || HEIGHT<=0){
		std::cerr<<"illegal width or height"<<std::endl;
		exit(1);
	}

	FTYPE *U_Curr = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	FTYPE *U_Next = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	init_domain(U_Curr, HEIGHT, WIDTH);
	init_domain(U_Next, HEIGHT, WIDTH);
	FTYPE *converge_sqd = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(WIDTH/H));

	cudaSetDevice(0);

	/*
	 * create gpu memory
	 */
	FTYPE *U_Curr_d;
	FTYPE *U_Next_d;
	checkCudaErr(cudaMalloc(&U_Curr_d, sizeof (FTYPE) * (int) floor(HEIGHT/H) *
									(int) floor (WIDTH / H)));
	checkCudaErr(cudaMalloc(&U_Next_d, sizeof (FTYPE) * (int) floor(HEIGHT/H) *
										(int) floor (WIDTH / H)));
	FTYPE *converge_sqd_d;
	checkCudaErr(cudaMalloc(&converge_sqd_d, sizeof(FTYPE) * (int)floor(WIDTH/H)));

	/*
	 * copy data in
	 */
	double t1, t2;
	t1 = getTime();
	checkCudaErr(
			cudaMemcpy(U_Curr_d, U_Curr, sizeof (FTYPE) * (int)floor(HEIGHT/H) *
						(int) floor (WIDTH / H), cudaMemcpyHostToDevice));
	t2 = getTime();
	double copyinTime = t2 - t1;

	/*
	 * main iterate computing
	 */
	int gridw = ((int) floor (WIDTH / H)+blockSize-1) / blockSize;
	int gridh = ((int) floor(HEIGHT/H) + blockSize-1)/blockSize;
	dim3 jacobiGrid(gridw, gridh, 1);
	dim3 jacobiBlock(blockSize, blockSize, 1);
	dim3 convergeGrid(gridw,1,1);
	dim3 convergeBlock(blockSize,1,1);

	double kernelrunTime=0;
	double hostcompTime = 0;
	double copyoutTime=0;
	int iters = 1;
	while(1){
		if(iters % 1000 ==0)
			std::cout<<"iteration: "<<iters<<" ..."<<std::endl;
		t1 = getTime();
		/* jacobi iterate */
		jacobi_kernel<<<jacobiGrid, jacobiBlock>>>(
				U_Curr_d, U_Next_d, (int)floor(HEIGHT/H), (int)floor(WIDTH/H));
		/*check if convergence */
		get_convergence_sqd_kernel<<<convergeGrid, convergeBlock>>>(
				U_Curr_d, U_Next_d, converge_sqd_d,
				(int)floor(HEIGHT/H), (int)floor(WIDTH/H));
		cudaDeviceSynchronize();
		t2 = getTime();
		kernelrunTime += t2-t1;
		/*
		 * copy data out
		 */
		t1=getTime();
		checkCudaErr(
				cudaMemcpy(converge_sqd, converge_sqd_d, sizeof(FTYPE)*(int)floor(WIDTH/H), cudaMemcpyDeviceToHost));
		t2=getTime();
		copyoutTime += t2-t1;

		t1 = getTime();
		double converge = get_convergence_sqd(converge_sqd, (int)floor(WIDTH/H));
		t2 = getTime();
		hostcompTime += t2-t1;
		if(sqrt(converge) <= EPSILON)
			break;
		FTYPE *tmp = U_Curr_d;
		U_Curr_d = U_Next_d;
		U_Next_d = tmp;
		iters++;
	}
	/*
	 * copy data out
	 */
	t1 = getTime();
	checkCudaErr(
			cudaMemcpy(U_Next, U_Next_d, sizeof (float) * (int)floor(HEIGHT/H) *
							(int) floor (WIDTH / H), cudaMemcpyDeviceToHost));
	t2 = getTime();
	copyoutTime += t2 -t1;

	free(U_Curr);
	free(U_Next);
	cudaFree(U_Curr_d);
	cudaFree(U_Next_d);
	cudaFree(converge_sqd_d);
	cudaDeviceReset();

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tDomain size: "<<WIDTH<<" X "<<HEIGHT<<std::endl;
		std::cout<<"\tAccuracy: "<<EPSILON<<std::endl;
		std::cout<<"\tIterations: "<<iters<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<copyinTime<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<copyoutTime<<"(s)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<kernelrunTime<<"(s)"<<std::endl;
		std::cout<<"\t\thost compute time: "<<hostcompTime<<"(s)"<<std::endl;
	}

}



