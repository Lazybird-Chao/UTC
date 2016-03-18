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
	void init(long loopN, unsigned seed, double range_lower, double range_upper,
			double *time,  double *ret)
	{
		if(getLrank()==0)
		{
			m_seed = seed;
			m_range_lower = range_lower;
			m_range_upper = range_upper;
			int total_nthreads = getGsize();
			int local_nthreads = getLsize();
			m_loopN = loopN/total_nthreads;
			m_res = (double*)malloc(sizeof(double)*local_nthreads);
			time_run_cost = time;
			m_ret = ret;
		}

	}

	void run()
	{
		double tmp_sum=0;
		double tmp_x=0;
		int my_localid = getLrank();
		int local_nthreads = getLsize();
		std::srand(m_seed);

		Timer timer;
		timer.start();
		double tmp_lower = m_range_lower;
		double tmp_upper=m_range_upper;
		unsigned int tmp_seed = m_seed;
		for(long i=0; i<m_loopN;i++)
		{
			tmp_x = tmp_lower+((double)rand_r(&tmp_seed)/RAND_MAX)*(tmp_upper-tmp_lower);
			tmp_sum+=f(tmp_x);
		}
		m_res[my_localid]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
		//timer.stop();
		//double t2 = timer.getThreadCpuTime();
		intra_Barrier();
		if(my_localid==0)
		{
			for(int i=1; i<local_nthreads;i++)
			{
				m_res[0]+=m_res[i];
			}
			m_res[0]/=local_nthreads;
			*m_ret = m_res[0];
			*time_run_cost = timer.stop();
			//std::cout<<"Result: "<<m_res[0]<<std::endl;
			//std::cout<<"cost time: "<<t<<std::endl;
		}
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
	double *m_res;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

	double *time_run_cost;
	double *m_ret;

};


int main(int argc, char*argv[])
{
	UtcContext ctx(argc, argv);
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
	Timer Timer;
	ctx.Barrier();
	Timer.start();
	Task<IntegralCaculator> integral_f(rlist);
	double runtime=0;
	double ret=0;
	integral_f.init(loopN, 1, 1.0, 10.0, &runtime, &ret);
	integral_f.run();
	integral_f.wait();
	ctx.Barrier();
	double totaltime = Timer.stop();

	double gret;
	MPI_Reduce(&ret, &gret, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	gret = gret/ctx.numProcs();
	for(int i=0; i<ctx.numProcs(); i++){
		if(ctx.getProcRank()==i){
			std::cout<<"Proc "<<i<<std::endl;
			std::cout<<"\tTotal time: "<<totaltime<<std::endl;
			std::cout<<"\trun() time: "<<runtime<<std::endl;
			std::cout<<"\tresult: "<<ret<<std::endl;
		}
		ctx.Barrier();
	}
	if(!ctx.getProcRank())
		std::cout<<"final result: "<<gret<<std::endl;
	return 0;
}

