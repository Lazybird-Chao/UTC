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
	void init(long loopN, unsigned seed, double range_lower, double range_upper, double *time)
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
		}

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
		double tmp_lower = m_range_lower;
		double tmp_upper=m_range_upper;
		unsigned int tmp_seed = m_seed;
		for(long i=0; i<m_loopN;i++)
		{
			tmp_x = tmp_lower+((double)rand_r(&tmp_seed)/RAND_MAX)*(tmp_upper-tmp_lower);
			tmp_sum+=f(tmp_x);
		}
		m_res[my_thread]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
		intra_Barrier();
		if(my_thread==0)
		{
			for(int i=1; i<local_nthreads;i++)
			{
				m_res[0]+=m_res[i];
			}
			m_res[0]/=local_nthreads;
			*time_run_cost = timer.stop();
		}

	}

private:
	long m_loopN;
	double *m_res;
	unsigned m_seed;
	double m_range_lower;
	double m_range_upper;

	double *time_run_cost;

};


int main(int argc, char*argv[])
{
	UtcContext ctx(argc, argv);
	/*
	 * run like ./a.out nthreads  nloops
	 */
	int nthreads = std::atoi(argv[1]);
	double runtime[32];
	long loopN = std::atol(argv[2]);

	Timer Timer;
	Timer.start();
	std::vector<Task<IntegralCaculator>*> taskArray;
	int numproc = ctx.numProcs();
	/* create nthreads*numproc tasks, each task run with one thread
	 *
	 */
	for(int i=0; i<numproc; i++){
		for(int j=0; j<nthreads; j++){
			ProcList rlist(i);
			taskArray.push_back(new Task<IntegralCaculator>(rlist));
		}
	}
	for(int i=0; i<numproc; i++){
		for(int j=0; j<nthreads; j++){
			taskArray[j+i*nthreads]->init(loopN, 1, 1.0, 10.0, &runtime[j]);
			taskArray[j+i*nthreads]->run();
		}
	}
	for(int i=0; i<numproc; i++){
			for(int j=0; j<nthreads; j++){
				taskArray[j+i*nthreads]->wait();
			}
	}
	double totaltime = Timer.stop();

	for(auto &i: taskArray){
		delete i;
	}
	taskArray.clear();
	for(int i=0; i<numproc; i++){
		if(ctx.getProcRank()==i){
			std::cout<<"Total time on proc "<<i<<": "<<totaltime<<std::endl;
			std::cout<<"run() time: "<<std::endl;
			for(int j=0; j<nthreads; j++){
				std::cout<<"\tTask "<<j+i*nthreads<<":"<<runtime[j]<<std::endl;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	return 0;
}

