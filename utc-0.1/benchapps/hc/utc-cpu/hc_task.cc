/*
 * hc_task.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */

#include "task.h"
#include <iostream>

#define H 1.0
#define T_SRC0 1550.0
#define ITERMAX 100

thread_local int HeatConductionWorker::thread_numRows;
thread_local int HeatConductionWorker::thread_startRowIndex;

void init_domain(FTYPE *domain_ptr, int h, int w){
	for (int j = 0; j < (int)floor(h/H); j++) {
		for (int i = 0; i < (int) floor (w / H); i++) {
			domain_ptr[j*((int) floor (w / H)) + i] = 0.0;
		}
	}
}

inline void enforce_bc_par(
		FTYPE *domain_ptr,
		int i,
		int j,
		int w,
		int h,
		int startRowIndex,
		int total_rows){
	if(i==(w/2-1) && j+startRowIndex==0){
		domain_ptr[j*w + i] = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		domain_ptr[j*w + i] = 0.0;
	}
}

//inline FTYPE get_var_par()  __attribute__((always_inline));
inline FTYPE get_var_par(
		FTYPE *domain_ptr,
		int i, //column
		int j, //row
		int w,
		int h_s,
		int h_e,
		FTYPE *top_row,
		FTYPE *bottom_row,
		int startRowIndex,
		int total_rows){

	FTYPE ret_val;
	if(i == w/2-1 && j+startRowIndex==0){
		ret_val = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		ret_val = 0.0;
	}
	else if(j+startRowIndex<h_s )
		ret_val = top_row[i];
	else if(j+startRowIndex>h_e-1)
		ret_val = bottom_row[i];
	else
		ret_val = domain_ptr[j*w + i];

	return ret_val;
}

inline FTYPE f(int i, int j){
	return 0.0;
}


void HeatConductionWorker::initImpl(int w, int h, FTYPE e, FTYPE *dmatrix){
	if(__localThreadId == 0){
		this->w = w;
		this->h = h;
		epsilon = e;
		domainMatrix = dmatrix;

		top_row.init(w);
		bottom_row.init(w);

		converge_sqd = new FTYPE[__numLocalThreads];
		top_ptr = new FTYPE[w];
		bot_ptr = new FTYPE[w];
		localCurr = (FTYPE*)malloc((h/__numGroupProcesses)*w*sizeof(FTYPE));
		localNext = (FTYPE*)malloc((h/__numGroupProcesses)*w*sizeof(FTYPE));
		process_numRows = h / __numGroupProcesses;
		process_startRowIndex = __processIdInGroup * process_numRows;

	}
	__fastIntraSync.wait();
	int rowsPerThread = process_numRows/__numLocalThreads;
	if(__localThreadId < process_numRows%__numLocalThreads){
		thread_numRows = rowsPerThread +1;
		thread_startRowIndex = __localThreadId*(rowsPerThread +1);
	}
	else{
		thread_numRows = rowsPerThread;
		thread_startRowIndex = __localThreadId*rowsPerThread + process_numRows%__numLocalThreads;
	}
	/*
	std::vector<int> cpus = getAffinity();
	msleep_for(__localThreadId*100);
	std::cout<<"thread: "<<__localThreadId<<": ";
	for(int i = 0; i < cpus.size(); i++)
		std::cout<<cpus[i]<<" ";
	std::cout<<std::endl;
	 */
	inter_Barrier();
	if(__globalThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void HeatConductionWorker::runImpl(double runtime[][MAX_TIMER], int *iteration){
	if(__globalThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double computetime = 0;
	double totaltime = 0;
	double commtime = 0;

	if(__localThreadId == 0){
		init_domain(localCurr, process_numRows, w);
		init_domain(localNext, process_numRows, w);
		init_domain(top_ptr, 1, w);
		init_domain(bot_ptr, 1, w);
	}

	int iters = 1;
	inter_Barrier();
	FTYPE *curr = localCurr + thread_startRowIndex*w;
	FTYPE *next = localNext + thread_startRowIndex*w;
	timer0.start();
	int h_s = process_startRowIndex;
	int h_e = process_startRowIndex+process_numRows;
	int s_s = process_startRowIndex+thread_startRowIndex;
	FTYPE up, down, left, right;
	while(iters <= ITERMAX){
		if(iters % 1000 ==0 && __globalThreadId == 0)
			std::cout<<"iter "<<iters<<"...\n";
		timer.start();
		for(int j = 0; j<thread_numRows; j++){
			for(int i = 0; i<w; i++){
				next[j*w+i] =
						0.25 * (
						get_var_par( curr, i - 1,j, w, h_s, h_e, top_ptr, bot_ptr, s_s, h)
					+ get_var_par ( curr, i + 1, j,w,h_s, h_e, top_ptr, bot_ptr, s_s, h)
					+ get_var_par ( curr, i, j-1 ,w, h_s, h_e, top_ptr, bot_ptr, s_s, h)
					+ get_var_par ( curr,i, j+1 ,w, h_s, h_e, top_ptr, bot_ptr, s_s, h));

				/*
				int ii, jj;
				ii = i-1;
				jj = j;
				if(ii == w/2-1 && jj+s_s==0){
					up = T_SRC0;
				}
				else if(ii<=0 || jj+s_s<=0 || ii>=w-1 || jj+s_s>=h-1){
					up = 0.0;
				}
				else if(jj+s_s<h_s )
					up = top_ptr[ii];
				else if(jj+s_s>h_e-1)
					up = bot_ptr[ii];
				else
					up = curr[jj*w + ii];
				//
				ii = i+1;
				jj = j;
				if(ii == w/2-1 && jj+s_s==0){
					down = T_SRC0;
				}
				else if(ii<=0 || jj+s_s<=0 || ii>=w-1 || jj+s_s>=h-1){
					down = 0.0;
				}
				else if(jj+s_s<h_s )
					down = top_ptr[ii];
				else if(jj+s_s>h_e-1)
					down = bot_ptr[ii];
				else
					down = curr[jj*w + ii];
				//
				ii = i;
				jj = j-1;
				if(ii == w/2-1 && jj+s_s==0){
					left = T_SRC0;
				}
				else if(ii<=0 || jj+s_s<=0 || ii>=w-1 || jj+s_s>=h-1){
					left = 0.0;
				}
				else if(jj+s_s<h_s )
					left = top_ptr[ii];
				else if(jj+s_s>h_e-1)
					left = bot_ptr[ii];
				else
					left = curr[jj*w + ii];
				//
				ii = i;
				jj = j+1;
				if(ii == w/2-1 && jj+s_s==0){
					right = T_SRC0;
				}
				else if(ii<=0 || jj+s_s<=0 || ii>=w-1 || jj+s_s>=h-1){
					right = 0.0;
				}
				else if(jj+s_s<h_s )
					right = top_ptr[ii];
				else if(jj+s_s>h_e-1)
					right = bot_ptr[ii];
				else
					right = curr[jj*w + ii];

				next[j*w + i] = 0.25*(up+down+left+right);
				*/

				enforce_bc_par(next, i, j, w, process_numRows, s_s, h);
			}

		}
		FTYPE sum = 0.0;
		for(int j = 0; j<thread_numRows; j++){
			for(int i = 0; i<w; i++){
				sum += (curr[j*w + i] - next[j*w + i])*
						(curr[j*w + i] - next[j*w + i]);
			}
		}
		computetime += timer.stop();
		converge_sqd[__localThreadId] = sum;
		__fastIntraSync.wait();
		if(__localThreadId == 0){
			top_row.storeblock(next, 0, w);
			bottom_row.storeblock(next+(process_numRows-1)*w, 0, w);
			total_converge = 0;
			for(int i = 1; i<__numLocalThreads; i++)
				converge_sqd[0] += converge_sqd[i];
		}
		/*
		 * reduce convergesqd value of each node and
		 * bcast result back to each node
		 */
		timer.start();
		TaskReduceSumBy<FTYPE, 0>(this, &converge_sqd[0], &total_converge, 1, 0);
		if(__globalThreadId == 0)
			total_converge = sqrt(total_converge);
		TaskBcastBy<FTYPE, 0>(this, &total_converge, 1, 0);
		__fastIntraSync.wait();
		commtime += timer.stop();
		if(total_converge <= epsilon)
			break;
		FTYPE *tmp = next;
		next = curr;
		curr = tmp;
		iters++;
		timer.start();
		/*
		 * fetch bottom and top from pre-node and next node
		 */
		if(__localThreadId == 0 && __processIdInGroup > 0){
			bottom_row.rloadblock(__processIdInGroup-1,top_ptr,0, w);
		}
		if(__localThreadId == __numLocalThreads-1 &&
				__processIdInGroup < __numGroupProcesses-1){
			top_row.rloadblock(__processIdInGroup+1, bot_ptr, 0, w);
		}
		__fastIntraSync.wait();
		commtime += timer.stop();
	}
	/*
	 * gather output to proc 0
	 */
	timer.start();
	TaskGatherBy<FTYPE, 0>(this, next, process_numRows*w,
								domainMatrix, process_numRows*w,
								0);
	__fastIntraSync.wait();
	commtime += timer.stop();

	inter_Barrier();
	totaltime = timer0.stop();
	if(__localThreadId == 0){
		delete top_ptr;
		delete bot_ptr;
		top_row.destroy();
		bottom_row.destroy();
		delete localCurr;
		delete localNext;
	}
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = commtime;
		if(__localThreadId == 0){
			*iteration = iters;
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
		}
	}
}

















