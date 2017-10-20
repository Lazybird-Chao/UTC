/*
 * task.h
 *
 *  Created on: Oct 4, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MC_UTC_CPU_TASK_H_
#define BENCHAPPS_MC_UTC_CPU_TASK_H_

#include "Utc.h"
class IntegralCaculator: public UserTaskBase
{
public:
	void initImpl(long loopN, unsigned seed, double range_lower, double range_upper);

	void runImpl(double runtime[][3]);


private:
	long m_loopN;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

	double m_res;

	double *m_local_res;

};




#endif /* BENCHAPPS_MC_UTC_CPU_TASK_H_ */
