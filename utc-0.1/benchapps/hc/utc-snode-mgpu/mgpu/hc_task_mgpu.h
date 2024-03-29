/*
 * hc_task_sgpu.h
 *
 *  Created on: Mar 23, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_HC_UTC_SGPU_HC_TASK_SGPU_H_
#define BENCHAPPS_HC_UTC_SGPU_HC_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "../typeconfig.h"

class hcMGPU: public UserTaskBase{
private:
	int w;
	int h;
	FTYPE epsilon;
	FTYPE *domainMatrix;

	static thread_local int local_numRows;
	static thread_local int local_startRowIndex;
	FTYPE *converge_sqd_array;
	FTYPE *top_row_array;
	FTYPE *bottom_row_array;

public:
	void initImpl(int w, int h, FTYPE e, FTYPE *dmatrix);

	void runImpl(double runtime[][5], int *iters, int blocksize, MemType memtype);
};



#endif /* BENCHAPPS_HC_UTC_SGPU_HC_TASK_SGPU_H_ */
