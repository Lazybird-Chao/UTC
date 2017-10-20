/*
 * task-s.cc
 *
 *  Created on: Oct 4, 2017
 *      Author: Chao
 */

/*
 * define the function f(x) that
 * whose integral is going to be
 * evaluated
 */

#include "task.h"

double f(double x)
{
	return 1.0/(x*x+1);
}



void IntegralCaculator::initImpl(long loopN, unsigned seed, double range_lower, double range_upper)
{
	if(__localThreadId==0)
	{
		m_seed = seed;
		m_range_lower = range_lower;
		m_range_upper = range_upper;
		m_loopN = loopN/__numGlobalThreads;
		m_res = 0;
		m_local_res = new double[__numLocalThreads];
	}

	inter_Barrier();

}

void IntegralCaculator::runImpl(double runtime[][3])
{
	double tmp_sum=0;
	double tmp_x=0;
	std::srand(m_seed);
	double *res_reduce = new double[__numLocalThreads];

	Timer timer0, timer;

	inter_Barrier();

	timer0.start();
	timer.start();
	double tmp_lower = m_range_lower;
	double tmp_upper=m_range_upper;
	unsigned int tmp_seed = m_seed;
	for(long i=0; i<m_loopN;i++)
	{
		tmp_x = tmp_lower+((double)rand_r(&tmp_seed)/RAND_MAX)*(tmp_upper-tmp_lower);
		tmp_sum+=f(tmp_x);
	}
	/*
	 __fastSpinLock.lock();
	m_res +=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
	__fastSpinLock.unlock();
	*/
	m_local_res[__localThreadId] = tmp_sum;
	__fastIntraSync.wait();
	runtime[__localThreadId][1] = timer.stop();
	timer.start();
	TaskReduceSumBy<double>(this, m_local_res, res_reduce, __numLocalThreads, 0);
	__fastIntraSync.wait();
	runtime[__localThreadId][2] = timer.stop();
	/*
	if(__localThreadId==0){
		res_gather = (double*)malloc(__numGroupProcesses*sizeof(double));
	}
	TaskGatherBy<double>(this, &m_res, 1, res_gather, 1, 0);
	*/
	double result = 0.0;
	if(__globalThreadId==0){
		for(int i=0; i<__numGlobalThreads; i++){
			result += res_reduce[i];
		}
		result /= __numGlobalThreads;
		std::cout<<result<<std::endl;
	}
	inter_Barrier();
	runtime[__localThreadId][0] = timer0.stop();
	/*
	if(__localThreadId==0){
		sleep_for(__processId);
		std::cout<<__processId<<": "<<__localThreadId<<": "<<
				__globalThreadId<<": "<<m_res[0]<<": "<<m_loopN<<std::endl;
	}
	inter_Barrier();
	sleep_for(my_thread);
	std::cout<<my_thread<<": "<<getpid()<<": "<<syscall(SYS_gettid)<<": "<<pthread_self()<<": "<<std::this_thread::get_id()
				<<"  "<<t2<<"  "<<m_loopN<<std::endl;
				*/
}




