/*
 * utc_shheat_image_gpu.cc
 *
 *  Created on: Nov 7, 2016
 *      Author: chao
 *
 *
 *      Single GPU version of heat image generation
 *
 */

#include "Utc.h"
#include "UtcGpu.h"
#include "../../helper_getopt.h"
#include "../../helper_printtime.h"

#include <iostream>

#include "utc_shheat_image_gpu_kernel.h"

using namespace iUtc;

#define DIM2(basetype, name, w1) basetype (*name)[w1]
#define STEPITER  1000

class HeatImageGPU:public UserTaskBase{
private:
	int mx;
	int my;
	int NITER;
	double delx=0.5;
	double dely=0.25;
	double rdx2, rdy2, beta;
	double *doublef;
	double *r;

	double *doublef_d;
	double *r_d;

	double *cmpres;

	int blocksize[2];

public:
	void initImpl(int dimx, int dimy, int iters, int blocksize[]){
		std::cout<<"begin init ..."<<std::endl;
		if(getUniqueExecution()){
			NITER = iters;
			mx = dimx;
			my = dimy;
			this->blocksize[0] = blocksize[0];
			this->blocksize[1] = blocksize[1];

			mx +=2;
			my +=2;
			rdx2 = 1/delx/delx;
			rdy2 = 1/delx/dely;
			beta = 1/(2*(rdx2+rdy2));
			doublef = (double*)malloc(2*mx*my*sizeof(double));
			r = (double*)malloc(mx*my*sizeof(double));

			//initialize data
			DIM2 (double, f, this->my) = (double(*)[this->my])doublef;
			DIM2 (double, newf, this->my) = f + mx;
			for(int i=0; i<mx; i++){
				for(int j=0; j<my; j++){
					if(i==0 || j==0 || i==mx-1 || j==my-1)
						newf[i][j]=f[i][j]=1.0;
					else
						newf[i][j]=f[i][j]=0.0;
					r[i*my+j] = 0.0;
				}
			}
			if(NITER < 100){
				cmpres = (double*)malloc(2*mx*my*sizeof(double));
				memcpy(cmpres, doublef, 2*mx*my*sizeof(double));
			}
			inter_Barrier();
			if(__localThreadId ==0){
				std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl on "
						<<__processId<<std::endl;
			}
		}
	}

	void runImpl(double *runtime){
		Timer t1, t2;
		std::cout<<"begin run ..."<<std::endl;

		int curf = 0;
		double *pf[2];
		double *f;
		double *newf;

		t1.start();
		checkCudaRuntimeErrors(
				cudaMalloc(&doublef_d, 2*mx*my*sizeof(double)));
		checkCudaRuntimeErrors(
				cudaMalloc(&r_d, mx*my*sizeof(double)));
		runtime[0] = t1.stop();
		pf[0] = doublef_d;
		pf[1] = pf[0] + mx*my;

		t1.start();
		t2.start();
		checkCudaRuntimeErrors(
				cudaMemcpy(doublef_d, doublef, 2*mx*my*sizeof(double),cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(
				cudaMemcpy(r_d, r, mx*my*sizeof(double), cudaMemcpyHostToDevice));
		runtime[1] = t1.stop();

		GpuKernel mykernel;
		// my is the column, mx is the row
		// in cuda, dim.x is colum, dim.y is row
		// use actual image size(mx-2, my-2) to set cuda grid
		mykernel.setGridDim((my-2 + blocksize[0]-1)/blocksize[0],
				(mx-2 + blocksize[1]-1)/blocksize[1]);
		mykernel.setBlockDim(blocksize[0],blocksize[1]);
		mykernel.setNumArgs(8);
		//mykernel.setArgs<double*>(0, pf[curf]);
		//mykernel.setArgs<double*>(1, pf[1-curf]);
		mykernel.setArgs<double*>(2, r_d);
		mykernel.setArgs<double>(3, rdx2);
		mykernel.setArgs<double>(4, rdy2);
		mykernel.setArgs<double>(5, beta);
		mykernel.setArgs<int>(6, mx);
		mykernel.setArgs<int>(7, my);

		for(int n=0; n < NITER; n++){
			if(__globalThreadId ==0){
				if(n%STEPITER ==0)
					std::cout<<"Iteration "<<n<<std::endl;
			}
			f = pf[curf];
			newf = pf[1-curf];
			mykernel.setArgs<double*>(0, f);
			mykernel.setArgs<double*>(1, newf);
			t1.start();
			mykernel.launchKernel((const void*)&heatImage_kernel);
			runtime[2] += t1.stop();
			curf = 1-curf;
		}

		t1.start();
		checkCudaRuntimeErrors(
				cudaMemcpy(doublef, doublef_d, 2*mx*my*sizeof(double), cudaMemcpyDeviceToHost));
		runtime[3] = t1.stop();
		runtime[4] = t2.stop();

		long err;
		if(NITER < 100){
			err = compareCompute(cmpres, 1e-9);
			if(err>0)
				std::cout<<"run error: "<<err<<std::endl;
			else
				std::cout<<"run correct !"<<std::endl;
		}


		// write to file
		char outfile[20] = "out.img";
		if (!__globalThreadId) {
			FILE*fp = fopen (outfile, "w");
			fclose (fp);

		}
		for (int j = 0; j < __numProcesses; j++) {
			inter_Barrier();
			if(getUniqueExecution()){
				if (j == __processId) {
					FILE* fp = fopen (outfile, "a");
					for (int i = 1; i < (mx - 1); i++)
						fwrite (&(doublef[mx*my + i*my + 1]), my - 2, sizeof (double), fp);
					fclose (fp);
				}
			}
		}

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl on "<<
					__processId<<".\n";
		}


	}

	long compareCompute(double *res, double eps = 1e-9){
		int curf = 0;
		double *pf[2];
		double *f;
		double *newf;
		pf[0] = res;
		pf[1] = res + mx*my;

		for(int n=0; n<NITER; n++){
			f = pf[curf];
			newf = pf[1-curf];
			for (int i = 1; i < mx-1; i++) {
				for (int j = 1; j < my-1; j++) {
					newf[i*my + j] =
						((f[(i - 1)*my + j] + f[(i + 1)*my + j]) * rdx2 +
						 (f[i*my + j - 1] + f[i*my + j + 1]) * rdy2 - r[i*my + j]) * beta;
				}
			}
			curf = 1-curf;
		}

		long err=0;
		for(int i=0; i< 2*mx*my; i++){
			if(fabs(res[i] - doublef[i]) > eps)
				err++;
		}
		return err;
	}

	~HeatImageGPU(){
		if(doublef)
			free(doublef);
		if(r)
			free(r);
		if(cmpres)
			free(cmpres);
		if(doublef_d)
			cudaFree(doublef_d);
		if(r_d)
			cudaFree(r_d);
	}

};

int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads=0;
	int nprocs=0;
	int dimx=600;
	int dimy=800;
	int iters=0;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:x:y:i:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case 'x':
			dimx = atoi(optarg);
			break;
		case 'y':
			dimy = atoi(optarg);
			break;
		case 'i':
			iters = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:x:y:i:");
	}
	if(iters==0)
		iters = 5000;
	int procs = ctx.numProcs();
	if(nprocs != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	int myproc = ctx.getProcRank();
	if(nthreads != 1){
		std::cerr<<"only run with 1 thread for this program\n";
		return 1;
	}

	double runtime[5];
	int blocksize[2] = {16,16};
	Task<HeatImageGPU> myHImage(ProcList(0), TaskType::gpu_task);
	myHImage.init(dimx, dimy, iters, blocksize);

	myHImage.run(runtime);

	myHImage.wait();
	myHImage.finish();

	if(myproc==0){
		std::cout<<"total time: "<<runtime[4]<<std::endl;
		std::cout<<"mem alloc time: "<<runtime[0]<<std::endl;
		std::cout<<"mem copyin time: "<<runtime[1]<<std::endl;
		std::cout<<"kernel run time: "<<runtime[2]<<std::endl;
		std::cout<<"mem copyout time: "<<runtime[3]<<std::endl;

		print_time(5, runtime);
	}


	return 0;
}



