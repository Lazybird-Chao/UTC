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

#include "task.h"

#define MAX_THREADS 64

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>


/* main UTC namespace */
using namespace iUtc;


int main(int argc, char*argv[])
{
	bool printTime = false;
	long loopN = 1;

	int nthreads=1;
	int nprocess=1;

	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	 while ( (opt=getopt(argc,argv,"a:o:i:n:bvt:p:m:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
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
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}

	std::vector<int> rank;
	int numproc = ctx.numProcs();
	for(int i=0; i<numproc; i++){
		for(int j=0; j<nthreads; j++)
			rank.push_back(i);
	}
	ProcList rlist(rank);	//create nthreads runing on each proc

	Task<IntegralCaculator> integral_f(rlist);
	double runtime_m[MAX_THREADS][2];
	integral_f.init(loopN, 1, 1.0, 10.0);
	integral_f.run(runtime_m);
	integral_f.wait();

	if(myproc == 0 && printTime){
		double runtime[2] = {0,0};
		for(int i =0; i<nthreads; i++)
			for(int j = 0; j<2; j++)
				runtime[j] += runtime_m[i][j];
		for(int j = 0; j<2; j++)
			runtime[j] /= nthreads;

		std::cout<<"Test complete !!!"<<std::endl;

		std::cout<<"\ttotal run time: "<<runtime[0]*1000<<std::endl;
		std::cout<<"\tcompute time: "<<runtime[1]*1000<<std::endl;
		for(int i=0; i<2; i++)
			runtime[i] *= 1000;
		print_time(2, runtime);
	}
	return 0;
}

