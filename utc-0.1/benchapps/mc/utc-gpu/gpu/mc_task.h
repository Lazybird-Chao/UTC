/*
 * mc_task.h
 *
 *  Created on: Oct 5, 2017
 *      Author: chaoliu
 */

#ifndef MC_TASK_H_
#define MC_TASK_H_

#include "Utc.h"
#include "UtcGpu.h"

class IntegralCaculator: public UserTaskBase
{
public:
	void initImpl(long loopN, unsigned int seed, double range_lower, double range_upper);

	void runImpl(double runtime[][5], int bsize, int gsize);


private:
	long m_loopN;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

	double m_res;

};



#endif /* MC_TASK_H_ */
