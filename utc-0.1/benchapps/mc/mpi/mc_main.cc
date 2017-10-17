/*
 * MC_Integral.cc
 *
 *  Using Monte Carlo method to compute
 *  one dimensional definite integral
 */



/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "mpi.h"
#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"


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
			m_loopN = loopN;;
			m_res = (double*)malloc(sizeof(double));


	}

	void run(double* runtime)
	{
		double tmp_sum=0;
		double tmp_x=0;
		std::srand(m_seed);
		int __numProcesses;
		int myproc;
		MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
		MPI_Comm_size(MPI_COMM_WORLD, &__numProcesses);

		double t1,t2;
		t1 = MPI_Wtime();
		double tmp_lower = m_range_lower;
		double tmp_upper=m_range_upper;
		unsigned int tmp_seed = m_seed;
		for(long i=0; i<m_loopN;i++)
		{
			tmp_x = tmp_lower+((double)rand_r(&tmp_seed)/RAND_MAX)*(tmp_upper-tmp_lower);
			tmp_sum+=f(tmp_x);
		}
		m_res[0]=tmp_sum*(m_range_upper-m_range_lower)/m_loopN;
		t2 = MPI_Wtime();
		//MPI_Barrier(MPI_COMM_WORLD);
		runtime[1] = t2-t1;

		double *res_gather;

		res_gather = (double*)malloc(__numProcesses*sizeof(double));
		t2 = MPI_Wime();
		MPI_Gather(m_res,sizeof(double), MPI_CHAR, res_gather,sizeof(double), MPI_CHAR,  0, MPI_COMM_WORLD);
		runtime[2] = MPI_Wtime() - t2;
		double result = 0.0;
		if(myproc==0){
			for(int i=0; i<__numProcesses; i++){
				result += res_gather[i];
			}
			result /= __numProcesses;
			//runtime[0] = timer0.stop();
			std::cout<<result<<std::endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		t2 = MPI_Wtime();
		runtime[0] = t2-t1;
		if(myproc==0){
		//	sleep_for(__processId);
		//	std::cout<<__processId<<": "<<__localThreadId<<": "<<
		//			__globalThreadId<<": "<<m_res[0]<<": "<<m_loopN<<std::endl;
		}
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
	bool	printTime = false;
	long loopN = std::atol(argv[1]);
	int nprocess = 1;


	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"n:vp:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'p': nprocess = atoi(optarg);
				  break;
			case 'n': loopN = atol(optarg);
					  break;
			case '?':
				break;
			default:
				break;
		}
	}
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}


	MPI_Init(&argc, &argv);
	/*
	 *  run like ./a.out  nthread   nloops
	 */
	int myproc;
	int procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	loopN /= nprocess;


	IntegralCaculator integral_f;
	double runtime[3] = {0,0,0};
	integral_f.init(loopN, 1, 1.0, 10.0);
	integral_f.run(runtime);
	double runtime_reduce[3] = {0,0,0};
	MPI_Reduce(runtime, runtime_reduce, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(myproc==0 && printTime){
		runtime[0] = runtime_reduce[0]/nprocess;
		runtime[1] = runtime_reduce[1]/nprocess;
		runtime[2] = runtime_reduce[2]/nprocess;
		std::cout<<"N: "<<loopN<<std::endl;
		std::cout<<"total run time: "<<runtime[0]*1000<<std::endl;
		std::cout<<"compute time: "<<runtime[1]*1000<<std::endl;
		std::cout<<"comm time: "<<runtime[2]*1000<<std::endl;

		for(int i=0; i<3; i++)
			runtime[i] *= 1000;
		print_time(3, runtime);
	}

	MPI_Finalize();

	return 0;
}

