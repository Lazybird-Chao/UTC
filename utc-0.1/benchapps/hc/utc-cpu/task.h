/*
 * task.h
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include "Utc.h"
#include "typeconfig.h"
#define MAX_TIMER 9


template<typename T>
class Output: public UserTaskBase{
public:
	void runImpl(T *buffer, int w, int h, char *ofile);
};


class HeatConductionWorker : public UserTaskBase{
private:
	int w;
	int h;
	FTYPE epsilon;
	FTYPE *domainMatrix;
	GlobalScopedData<FTYPE> top_row;
	GlobalScopedData<FTYPE> bottom_row;

	FTYPE *top_ptr;
	FTYPE *bot_ptr;
	FTYPE *converge_sqd;
	FTYPE total_converge;
	FTYPE *localCurr;
	FTYPE *localNext;
	int process_numRows;
	int process_startRowIndex;

	static thread_local int thread_numRows;
	static thread_local int thread_startRowIndex;

public:
	void initImpl(int w, int h, FTYPE e, FTYPE *dmatrix);

	void runImpl(double runtime[][MAX_TIMER], int *iters);
};

#endif /* TASK_H_ */
