/*
 * hc_task.h
 *
 *  Created on: Oct 12, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_HC_UTC_GPU_GPU_HC_TASK_H_
#define BENCHAPPS_HC_UTC_GPU_GPU_HC_TASK_H_

#include "Utc.h"
#include "UtcGpu.h"
#include "../typeconfig.h"
#define MAX_TIMER 9

class HeatConductionWorker : public UserTaskBase{
private:
	int w;
	int h;
	FTYPE epsilon;
	FTYPE *domainMatrix;
	GlobalScopedData<FTYPE> top_row;
	GlobalScopedData<FTYPE> bottom_row;

	//FTYPE *top_ptr;
	//FTYPE *bot_ptr;
	FTYPE converge_sqd;
	FTYPE total_converge;
	//FTYPE *localCurr;
	//FTYPE *localNext;
	int process_numRows;
	int process_startRowIndex;


public:
	void initImpl(int w, int h, FTYPE e, FTYPE *dmatrix);

	void runImpl(double runtime[][MAX_TIMER], int *iters, int blockSize);
};



#endif /* BENCHAPPS_HC_UTC_GPU_GPU_HC_TASK_H_ */
