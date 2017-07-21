/*
 * hc_task_sgpu.cu
 *
 *  Created on: Mar 23, 2017
 *      Author: Chao
 */

#include "hc_task_mgpu.h"
#include "hc_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>



#define H 1.0
#define T_SRC0 550.0

thread_local int hcMGPU::local_numRows;
thread_local int hcMGPU::local_startRowIndex;

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


void hcMGPU::initImpl(int w, int h, FTYPE e, FTYPE*dmatrix){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->w = w;
		this->h = h;
		epsilon = e;
		domainMatrix = dmatrix;

		converge_sqd_array = new FTYPE[__numLocalThreads];
		top_row_array = new FTYPE[__numLocalThreads*w];
		bottom_row_array = new FTYPE[__numLocalThreads*w];
	}
	intra_Barrier();
	int rowsPerThread = h/__numLocalThreads;
	if(__localThreadId < h%__numLocalThreads){
		local_numRows = rowsPerThread +1;
		local_startRowIndex = __localThreadId*(rowsPerThread +1);
	}
	else{
		local_numRows = rowsPerThread;
		local_startRowIndex = __localThreadId*rowsPerThread + h%__numLocalThreads;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}


void hcMGPU::runImpl(double runtime[][5], int *iteration, int blockSize, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	GpuData<FTYPE> local_U_Curr((local_numRows)*(int)floor(w/H), memtype);
	GpuData<FTYPE> local_U_Next((local_numRows)*(int)floor(w/H), memtype);
	GpuData<FTYPE> converge_sqd((int)floor(w/H), memtype);
	init_domain(local_U_Curr.getH(true), local_numRows, w);

	GpuData<FTYPE> top_row((int)floor(w/H), memtype);
	GpuData<FTYPE> bottom_row((int)floor(w/H), memtype);
	init_domain(top_row.getH(true), 1, w);
	init_domain(bottom_row.getH(true), 1, w);

	//
	timer0.start();
	timer.start();
	local_U_Curr.sync();
	top_row.sync();
	bottom_row.sync();
	double copyinTime = timer.stop();

	//
	int gridw = (w+blockSize-1)/blockSize;
	int gridh = (local_numRows+blockSize-1)/blockSize;
	dim3 jacobiGrid(gridw, gridh, 1);
	dim3 jacobiBlock(blockSize, blockSize, 1);
	dim3 convergeBlock(blockSize*blockSize,1,1);
	dim3 convergeGrid(((int)floor (w / H)+convergeBlock.x-1)/convergeBlock.x,1,1);

	double kernelTime=0;
	double hostcompTime = 0;
	double copyoutTime=0;
	int iters = 1;

	while(1){
		//if(iters % 1000 ==0)
		//	std::cout<<"iteration: "<<iters<<" ..."<<std::endl;
		timer.start();
		if(iters % 2 ==1){
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					local_U_Curr.getD(), local_U_Next.getD(true),
					local_numRows, (int)floor(w/H),
					top_row.getD(), bottom_row.getD(), local_startRowIndex, h);
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					local_U_Curr.getD(), local_U_Next.getD(), converge_sqd.getD(true),
					local_numRows, (int)floor(w/H));
		}
		else{
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					local_U_Next.getD(), local_U_Curr.getD(true),
					local_numRows, (int)floor(w/H),
					top_row.getD(), bottom_row.getD(), local_startRowIndex, h);
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					local_U_Next.getD(), local_U_Curr.getD(), converge_sqd.getD(true),
					local_numRows, (int)floor(w/H));
		}
		checkCudaErr(cudaStreamSynchronize(__streamId));
		//cudaDeviceSynchronize();
		checkCudaErr(cudaGetLastError());
		kernelTime += timer.stop();

		timer.start();
		converge_sqd.sync();
		copyoutTime += timer.stop();

		timer.start();
		FTYPE converge = get_convergence_sqd(converge_sqd.getH(), (int)floor(w/H));
		converge_sqd_array[__localThreadId] = converge;
		intra_Barrier();
		if(getUniqueExecution()){
			for(int i=1; i<__numLocalThreads; i++)
				converge_sqd_array[0]+= converge_sqd_array[i];
		}
		hostcompTime += timer.stop();

		timer.start();
		if(__localThreadId>0){
			if(iters % 2 ==1)
				local_U_Next.fetchD(bottom_row_array+(__localThreadId-1)*w, 0, w*sizeof(FTYPE));
			else
				local_U_Curr.fetchD(bottom_row_array+(__localThreadId-1)*w, 0, w*sizeof(FTYPE));
		}
		if(__localThreadId<__numLocalThreads-1){
			if(iters%2==1)
				local_U_Next.fetchD(top_row_array+(__localThreadId+1)*w, (local_numRows-1)*w, w*sizeof(FTYPE));
			else
				local_U_Curr.fetchD(top_row_array+(__localThreadId+1)*w, (local_numRows-1)*w, w*sizeof(FTYPE));
		}
		copyoutTime += timer.stop();
		intra_Barrier();

		converge = converge_sqd_array[0];
		if(sqrt(converge) <= epsilon)
			break;
		iters++;

		timer.start();
		if(__localThreadId<__numLocalThreads-1)
			bottom_row.putD(bottom_row_array+__localThreadId*w);
		if(__localThreadId >0)
			top_row.putD(top_row_array+__localThreadId*w);
		copyinTime += timer.stop();
	}

	//*iteration = iters;
	timer.start();
	if(iters % 2 ==1){
		local_U_Next.sync();
		copyoutTime += timer.stop();
		totaltime = timer0.stop();
		local_U_Next.fetch(domainMatrix+local_startRowIndex*w);
	}
	else{
		local_U_Curr.sync();
		copyoutTime += timer.stop();
		totaltime = timer0.stop();
		local_U_Curr.fetch(domainMatrix+local_startRowIndex*w);
	}
	intra_Barrier();
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1]= kernelTime;
	runtime[__localThreadId][2]= copyinTime;
	runtime[__localThreadId][3]= copyoutTime;
	runtime[__localThreadId][4]= hostcompTime;

	if(__localThreadId ==0){
		*iteration = iters;
		delete converge_sqd_array;
		delete top_row_array;
		delete bottom_row_array;

		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}








