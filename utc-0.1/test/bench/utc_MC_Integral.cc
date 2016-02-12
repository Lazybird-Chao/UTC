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


#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/times.h>
#include <pthread.h>
#include <time.h>

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


class IntegralCaculator
{
public:
	void init(long loopN, unsigned seed, double range_lower, double range_upper)
	{
		if(getTrank()==0)
		{
			m_seed = seed;
			m_range_lower = range_lower;
			m_range_upper = range_upper;
			int total_nthreads = getGsize();
			int local_nthreads = getLsize();
			m_loopN = loopN/total_nthreads;
			m_res = (double*)malloc(sizeof(double)*local_nthreads);
		}

	}

	void run()
	{
		double tmp_sum=0;
		double tmp_x=0;
		int my_thread = getTrank();
		int local_nthreads = getLsize();
		std::srand(m_seed);

		/*sleep(my_thread);
		std::cout<<my_thread<<": "<<getpid()<<": "<<syscall(SYS_gettid)<<endl;
		// check thread affinity
		int s;
		cpu_set_t cpuset;
		pthread_t thread;
		CPU_ZERO(&cpuset);
		thread=pthread_self();
		s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		if(s==0)
		{
			std::cout<<"cpu_set: ";
			for (int j = 0; j < CPU_SETSIZE; j++)
				   if(CPU_ISSET(j, &cpuset))
					   printf("%d ", j);
			std::cout<<std::endl;
		}
		//change and set new affinity
		CPU_ZERO(&cpuset);
		CPU_SET(my_thread, &cpuset);
		pthread_setaffinity_np(thread, sizeof(cpu_set_t),&cpuset);
		intra_Barrier();*/

		Timer timer;
		timer.start();

		/*struct timespec st1, st2;
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &st1);
		struct tms cst1, cst2;
		times(&cst1);*/

		/*int c1, c2;
		c1 = c2= sched_getcpu();
		long count=0;*/
		double tmp_lower = m_range_lower;
		double tmp_upper=m_range_upper;
		unsigned int tmp_seed = m_seed;
		for(long i=0; i<m_loopN;i++)
		{
			tmp_x = tmp_lower+((double)rand_r(&tmp_seed)/RAND_MAX)*(tmp_upper-tmp_lower);
			tmp_sum+=f(tmp_x);

				/*c2=sched_getcpu();
				if(c2!=c1)
				{
					c1 = c2;
					count++;
				}*/

		}
		m_res[my_thread]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;

		/*clock_gettime(CLOCK_THREAD_CPUTIME_ID,&st2);
		double diff = (st2.tv_sec - st1.tv_sec) + (st2.tv_nsec - st1.tv_nsec)/1e9;
		times(&cst2);
		clock_t utime = cst2.tms_utime-cst1.tms_utime;
		clock_t systime = cst2.tms_stime - cst1.tms_stime;*/

		timer.stop();
		//double t2 = timer.getThreadCpuTime();
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


		/*sleep_for(my_thread);
		std::cout<<my_thread<<": "<<getpid()<<": "<<syscall(SYS_gettid)<<": "<<pthread_self()<<": "<<std::this_thread::get_id()
					<<"  "<<t2<<"  "<<m_loopN<<std::endl;
					*/
		/*
		// check thread affinity
		CPU_ZERO(&cpuset);
		thread=pthread_self();
		s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		if(s==0)
		{
			std::cout<<"cpu_set: ";
			for (int j = 0; j < CPU_SETSIZE; j++)
				   if(CPU_ISSET(j, &cpuset))
					   printf("%d ", j);
			std::cout<<std::endl;
		}
		std::cout<<count<<std::endl;
		std::cout<<diff<<std::endl;
		std::cout<<utime<<"  "<<systime<<std::endl;*/
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

	/*std::cout<<"Total cpus: "<<std::thread::hardware_concurrency()<<std::endl;

	std::cout<<"main proc id: "<<getpid()<<": "<<syscall(SYS_gettid)<<"  thread id:"<<std::this_thread::get_id()<<std::endl;
	// check thread affinity
	int s;
	cpu_set_t cpuset;
	pthread_t thread;
	CPU_ZERO(&cpuset);
	thread=pthread_self();
	s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	if(s==0)
	{
		std::cout<<"cpu_set: ";
		for (int j = 0; j < CPU_SETSIZE; j++)
			   if(CPU_ISSET(j, &cpuset))
				   printf("%d ", j);
		std::cout<<std::endl;
	}*/

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

