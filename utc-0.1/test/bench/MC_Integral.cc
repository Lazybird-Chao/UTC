/*
 * MC_Integral.cc
 *
 *  Using Monte Carlo method to compute
 *  one dimensional definite integral
 */


/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

/* main UTC namespace */
using namespace iUtc;

#define USE_THREAD

/*
 * define the function f(x) that
 * whose integral is going to be
 * evaluated
 */
double f(double x)
{
	return 1.0/(x*x+1);
}


class IntegralCaculator
{
public:
	void init(long loopN, unsigned seed, double range_lower, double range_upper)
	{
		m_seed = seed;
		m_range_lower = range_lower;
		m_range_upper = range_upper;
		int total_nthreads = getGsize();
		int local_nthreads = getLsize();
		m_loopN = loopN/total_nthreads;
		m_res = (double*)malloc(sizeof(double)*local_nthreads);

	}

	void run()
	{
		double tmp_sum=0;
		double tmp_x=0;
		int my_thread = getTrank();
		int local_nthreads = getLsize();
		std::srand(m_seed);
		Timer timer;
		timer.start();
		for(long i=0; i<m_loopN;i++)
		{
			tmp_x = m_range_lower+((double)rand()/RAND_MAX)*(m_range_upper-m_range_lower);
			tmp_sum+=f(tmp_x);
		}
		m_res[my_thread]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
		double t2 = timer.stop();
		intra_Barrier();
		if(my_thread==0)
		{
			for(int i=1; i<local_nthreads;i++)
			{
				m_res[0]+=m_res[i];
			}
			m_res[0]/=local_nthreads;
			double t = timer.stop();
			std::cout<<"Result: "<<m_res[0]<<std::endl;
			std::cout<<"cost time: "<<t<<std::endl;
		}
		sleep_for(my_thread);
		std::cout<<my_thread<<": "<<getpid()<<": "<<pthread_self()<<": "<<std::this_thread::get_id()
					<<"  "<<t2<<"  "<<m_loopN<<std::endl;
	}

private:
	long m_loopN;
	double *m_res;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

};


int main(int argc, char*argv[])
{
	UtcContext ctx(argc, argv);

	int nthreads = std::atoi(argv[1]);
	//std::cout<<nthreads<<std::endl;
#ifdef USE_PROCESS
	int rank[16];
	for(int i=0;i<16;i++)
		rank[i]=i;
	RankList rlist(nthreads, rank);
#endif
#ifdef USE_THREAD
	RankList rlist(nthreads, 0);	//create nthreads runing on proc 0
#endif
	Task<IntegralCaculator> integral_f(rlist);

	//long loopN = 1000000000; //e9
	long loopN = std::atol(argv[2]);
	integral_f.init(loopN, 8, 1.0, 10.0);

	integral_f.run();

	integral_f.waitTillDone();

	return 0;
}

