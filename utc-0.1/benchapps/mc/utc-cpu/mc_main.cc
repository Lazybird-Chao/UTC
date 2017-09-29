/*
 * MC_Integral.cc
 *
 *  Using Monte Carlo method to compute
 *  one dimensional definite integral
 */


/* main UTC header file */
#include "Utc.h"
#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"


/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>


/* main UTC namespace */
using namespace iUtc;

#define USE_THREAD
//#define USE_PROCESS

/*
 * define the function f(x) that
 * whose integral is going to be
 * evaluated
 */
double f(double x)
{
	return 1.0/(x*x+1);
}


class IntegralCaculator: public UserTaskBase
{
public:
	void initImpl(long loopN, unsigned seed, double range_lower, double range_upper)
	{
		if(__localThreadId==0)
		{
			m_seed = seed;
			m_range_lower = range_lower;
			m_range_upper = range_upper;
			m_loopN = loopN/__numGlobalThreads;
			m_res = (double*)malloc(sizeof(double)*__numLocalThreads);
		}

	}

	void runImpl(double* runtime)
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
		m_res[__localThreadId]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
		intra_Barrier();
		if(__localThreadId==0)
		{
			for(int i=1; i<__numLocalThreads;i++)
			{
				m_res[0]+=m_res[i];
			}
			m_res[0]/=__numLocalThreads;
			runtime[1] = timer.stop();
		}
		intra_Barrier();
		double *res_gather;
		if(__localThreadId==0){
			//runtime[1] = timer.stop();
			res_gather = (double*)malloc(__numProcesses*sizeof(double));
		}
		SharedDataGather(m_res,sizeof(double), res_gather, 0);
		double result = 0.0;
		if(__globalThreadId==0){
			for(int i=0; i<__numProcesses; i++){
				result += res_gather[i];
			}
			result /= __numProcesses;
			//runtime[0] = timer0.stop();
			//std::cout<<result<<std::endl;
		}
		inter_Barrier();
		if(__localThreadId==0){
			runtime[0] = timer0.stop();
		//	sleep_for(__processId);
		//	std::cout<<__processId<<": "<<__localThreadId<<": "<<
		//			__globalThreadId<<": "<<m_res[0]<<": "<<m_loopN<<std::endl;
		}
		inter_Barrier();
		/*sleep_for(my_thread);
		std::cout<<my_thread<<": "<<getpid()<<": "<<syscall(SYS_gettid)<<": "<<pthread_self()<<": "<<std::this_thread::get_id()
					<<"  "<<t2<<"  "<<m_loopN<<std::endl;*/
	}

	~IntegralCaculator(){
		if(!m_res)
			free(m_res);
	}

private:
	long m_loopN;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

	double *m_res;



};


int main(int argc, char*argv[])
{
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	/*
	 *  run like ./a.out  nthread   nloops
	 */
	int nthreads = std::atoi(argv[1]);
	long loopN = std::atol(argv[2]);
#ifdef USE_THREAD
	std::vector<int> rank;
	int numproc = ctx.numProcs();
	for(int i=0; i<numproc; i++){
		for(int j=0; j<nthreads; j++)
			rank.push_back(i);
	}
	ProcList rlist(rank);	//create nthreads runing on each proc
#endif

	//
	//Task<IntegralCaculator> dumm_task(rlist);
	//

	Timer Timer;
	ctx.Barrier();
	Timer.start();
	Task<IntegralCaculator> integral_f(rlist);
	double runtime[2] = {0,0};
	integral_f.init(loopN, 1, 1.0, 10.0);
	integral_f.run(runtime);
	integral_f.wait();
	ctx.Barrier();
	double totaltime = Timer.stop();

	if(ctx.getProcRank()==0){
		std::cout<<"total run time: "<<runtime[0]*1000<<std::endl;
		std::cout<<"compute time: "<<runtime[1]*1000<<std::endl;
		for(int i=0; i<2; i++)
			runtime[i] *= 1000;
		print_time(2, runtime);
	}

	return 0;
}

