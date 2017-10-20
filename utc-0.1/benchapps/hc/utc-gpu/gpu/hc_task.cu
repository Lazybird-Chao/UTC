/*
 * hc_task.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */

#include "hc_task.h"
#include "hc_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>

#define H 1.0
#define T_SRC0 1550.0
#define ITERMAX 100

void init_domain(FTYPE *domain_ptr, int h, int w){
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


void HeatConductionWorker::initImpl(int w, int h, FTYPE e, FTYPE *dmatrix){
	if(__localThreadId == 0){
		this->w = w;
		this->h = h;
		epsilon = e;
		domainMatrix = dmatrix;

		top_row.init(w);
		bottom_row.init(w);

		converge_sqd = 0.0;
		//top_ptr = new FTYPE[w];
		//bot_ptr = new FTYPE[w];
		//localCurr = new FTYPE[h / __numGroupProcesses * w];
		//localNext = new FTYPE[h / __numGroupProcesses * w];
		process_numRows = h / __numGroupProcesses;
		process_startRowIndex = __processIdInGroup * process_numRows;

	}
	__fastIntraSync.wait();
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void HeatConductionWorker::runImpl(double runtime[][MAX_TIMER], int *iteration, int blockSize){
	if(__globalThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double computetime = 0;
	double totaltime = 0;
	double commtime = 0;
	double copytime = 0;

	//if(__localThreadId == 0){ //we only allow one thread in this proc
	GpuData<FTYPE> glocal_U_Curr((process_numRows)*(int)floor(w/H));
	GpuData<FTYPE> glocal_U_Next((process_numRows)*(int)floor(w/H));
	GpuData<FTYPE> gconverge_sqd((int)floor(w/H));
	init_domain(glocal_U_Curr.getH(true), process_numRows, w);
	GpuData<FTYPE> gtop_row((int)floor(w/H));
	GpuData<FTYPE> gbottom_row((int)floor(w/H));
	init_domain(gtop_row.getH(true), 1, w);
	init_domain(gbottom_row.getH(true), 1, w);

	int iters = 1;
	int gridw = (w+blockSize-1)/blockSize;
	int gridh = (process_numRows+blockSize-1)/blockSize;
	dim3 jacobiGrid(gridw, gridh, 1);
	dim3 jacobiBlock(blockSize, blockSize, 1);
	dim3 convergeBlock(blockSize*blockSize,1,1);
	dim3 convergeGrid(((int)floor (w / H)+convergeBlock.x-1)/convergeBlock.x,1,1);
	inter_Barrier();
	timer0.start();
	timer.start();
	glocal_U_Curr.sync();
	gtop_row.sync();
	gbottom_row.sync();
	copytime += timer.stop();
	while(iters <= ITERMAX){
		if(iters % 1000 ==0 && __globalThreadId == 0)
			std::cout<<"iter "<<iters<<"...\n";
		timer.start();
		if(iters % 2 ==1){
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					glocal_U_Curr.getD(), glocal_U_Next.getD(true),
					process_numRows, (int)floor(w/H),
					gtop_row.getD(), gbottom_row.getD(), process_startRowIndex, h);
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					glocal_U_Curr.getD(), glocal_U_Next.getD(), gconverge_sqd.getD(true),
					process_numRows, (int)floor(w/H));
		} else{
			jacobi_kernel<<<jacobiGrid, jacobiBlock,0,__streamId>>>(
					glocal_U_Next.getD(), glocal_U_Curr.getD(true),
					process_numRows, (int)floor(w/H),
					gtop_row.getD(), gbottom_row.getD(), process_startRowIndex, h);
			get_convergence_sqd_kernel<<<convergeGrid, convergeBlock, 0, __streamId>>>(
					glocal_U_Next.getD(), glocal_U_Curr.getD(), gconverge_sqd.getD(true),
					process_numRows, (int)floor(w/H));
		}
		checkCudaErr(cudaStreamSynchronize(__streamId));
		checkCudaErr(cudaGetLastError());
		computetime += timer.stop();

		timer.start();
		gconverge_sqd.sync();
		if(iters % 2 ==1)
			glocal_U_Next.fetchD(top_row.getPtr(), 0, w*sizeof(FTYPE));
		else
			glocal_U_Curr.fetchD(top_row.getPtr(), 0, w*sizeof(FTYPE));
		if(iters%2==1)
			glocal_U_Next.fetchD(bottom_row.getPtr(), (process_numRows-1)*w, w*sizeof(FTYPE));
		else
			glocal_U_Curr.fetchD(bottom_row.getPtr(), (process_numRows-1)*w, w*sizeof(FTYPE));
		copytime += timer.stop();
		timer.start();
		converge_sqd = get_convergence_sqd(gconverge_sqd.getH(), (int)floor(w/H));
		computetime += timer.stop();
		/*
		 * reduce convergesqd value of each node and
		 * bcast result back to each node
		 */
		timer.start();

		TaskReduceSumBy<FTYPE, 0>(this, &converge_sqd, &total_converge, 1, 0);
		total_converge = sqrt(total_converge);
		TaskBcastBy<FTYPE, 0>(this, &total_converge, 1, 0);

		commtime += timer.stop();
		if(total_converge <= epsilon)
			break;
		iters++;
		timer.start();
		/*
		 * fetch bottom and top from pre-node and next node
		 */
		if(__localThreadId == 0 && __processIdInGroup > 0){
			bottom_row.rloadblock(__processIdInGroup-1,gtop_row.getH(true),0, w);
			//gtop_row.sync();
		}
		if(__localThreadId == __numLocalThreads-1 &&
				__processIdInGroup < __numGroupProcesses-1){
			top_row.rloadblock(__processIdInGroup+1, gbottom_row.getH(true), 0, w);
			//gbottom_row.sync();
		}
		commtime += timer.stop();
		timer.start();
		gtop_row.sync();
		gbottom_row.sync();
		copytime += timer.stop();

	}
	/*
	 * gather output to proc 0
	 */
	timer.start();
	if(iters % 2 == 1){
		glocal_U_Next.sync();
	}else{
		glocal_U_Curr.sync();
	}
	copytime += timer.stop();
	timer.start();
	if(iters % 2 == 1){
		//glocal_U_Next.sync();
		TaskGatherBy<FTYPE, 0>(this, glocal_U_Next.getH(), process_numRows*w,
								domainMatrix, process_numRows*w,
								0);
	}else{
		//glocal_U_Curr.sync();
		TaskGatherBy<FTYPE, 0>(this, glocal_U_Curr.getH(), process_numRows*w,
								domainMatrix, process_numRows*w,
								0);
	}
	commtime += timer.stop();

	inter_Barrier();
	totaltime = timer0.stop();
	if(__localThreadId == 0){
		top_row.destroy();
		bottom_row.destroy();
	}
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = commtime;
		runtime[__localThreadId][3] = copytime;
		if(__localThreadId == 0){
			*iteration = iters;
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
		}
	}
}

















