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
	}

}

void IntegralCaculator::runImpl(double runtime[][2])
{
	double tmp_sum=0;
	double tmp_x=0;
	std::srand(m_seed);

	Timer timer0, timer;
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
	__fastSpinLock.lock();
	m_res +=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
	__fastSpinLock.unlock();
	runtime[__localThreadId][1] = timer.stop();
	//intra_Barrier();
	__fastIntraSync.wait();
	double *res_gather;
	if(__localThreadId==0){
		res_gather = (double*)malloc(__numGroupProcesses*sizeof(double));
	}
	TaskGatherBy<double>(this, &m_res, 1, res_gather, 1, 0);
	double result = 0.0;
	if(__globalThreadId==0){
		for(int i=0; i<__numGroupProcesses; i++){
			result += res_gather[i];
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




