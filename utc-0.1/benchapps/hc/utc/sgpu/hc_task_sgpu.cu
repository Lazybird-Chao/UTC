/*
 * hc_task_sgpu.cu
 *
 *  Created on: Mar 23, 2017
 *      Author: Chao
 */

#include "hc_task_sgpu.h"
#include "hc_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>



#define H 1.0
#define T_SRC0 550.0

template<typename T>
void init_domain(T *domain_ptr, int h, int w){
	for (int j = 0; j < (int)floor(h/H); j++) {
		for (int i = 0; i < (int) floor (w / H); i++) {
			domain_ptr[j*((int) floor (w / H)) + i] = 0.0;
		}
	}
}

template<typename T>
T get_convergence_sqd(T *sqd_array, int w){
	T sum = 0.0;
	for(int i=0; i< w; i++)
		sum += sqd_array[i];
	return sum;
}


void hcSGPU::initImpl(int w, int h, FTYPE e, FTYPE*dmatrix){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->w = w;
		this->h = h;
		epsilon = e;
		domainMatrix = dmatrix;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}


void hcSGPU::runImpl(double *runtime, int *iteration, int blockSize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	GpuData<FTYPE> U_Curr((int)floor(h/H)*(int)floor(w/H), memtype);
	GpuData<FTYPE> U_Next((int)floor(h/H)*(int)floor(w/H), memtype);
	GpuData<FTYPE> converge_sqd((int)floor(w/H), memtype);
	init_domain(U_Curr.getH(true), h, w);

	//
	timer.start();
	U_Curr.sync();
	double copyinTime = timer.stop();

	//
	int gridw = ((int) floor (w / H)+blockSize-1) / blockSize;
	int gridh = ((int) floor(h/H) + blockSize-1)/blockSize;
	dim3 jacobiGrid(gridw, gridh, 1);
	dim3 jacobiBlock(blockSize, blockSize, 1);
	dim3 convergeBlock(blockSize*blockSize,1,1);
	dim3 convergeGrid(((int)floor (w / H)+convergeBlock.x-1)/convergeBlock.x,1,1);

	double kernelTime=0;
	double hostcompTime = 0;
	double copyoutTime=0;
	int iters = 1;

	while(1){
		if(iters % 1000 ==0)
			std::cout<<"iteration: "<<iters<<" ..."<<std::endl;
		timer.start();
		if(iters % 2 ==1){
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					U_Curr.getD(), U_Next.getD(true), (int)floor(h/H), (int)floor(w/H));
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					U_Curr.getD(), U_Next.getD(), converge_sqd.getD(true),
					(int)floor(h/H), (int)floor(w/H));
		}
		else{
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					U_Next.getD(), U_Curr.getD(true), (int)floor(h/H), (int)floor(w/H));
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					U_Next.getD(), U_Curr.getD(), converge_sqd.getD(true),
					(int)floor(h/H), (int)floor(w/H));
		}
		checkCudaErr(cudaStreamSynchronize(__streamId));
		//cudaDeviceSynchronize();
		checkCudaErr(cudaGetLastError());
		kernelTime += timer.stop();

		timer.start();
		converge_sqd.sync();
		copyoutTime += timer.stop();

		timer.start();
		double converge = get_convergence_sqd(converge_sqd.getH(), (int)floor(w/H));
		hostcompTime += timer.stop();
		if(sqrt(converge) <= epsilon)
			break;
		iters++;
	}
	*iteration = iters;
	timer.start();
	if(iters % 2 ==1)
		U_Next.fetch(domainMatrix);
	else
		U_Curr.fetch(domainMatrix);
	copyoutTime += timer.stop();

	runtime[0] = copyinTime + copyoutTime + kernelTime + hostcompTime;
	runtime[1]= kernelTime;
	runtime[2]= copyinTime;
	runtime[3]= copyoutTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}








