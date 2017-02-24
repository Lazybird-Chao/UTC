/*
 * micro_main.cu
 *
 *      Author: Chao
 *
 * The micro test to simulate different gpu memory usage for compute bond
 * and memory access bond application.
 *
 *
 */

#include "../../common/helper_err.h"
#include "../../common/helper_getopt.h"
#include "Utc.h"
#include "UtcGpu.h"
#include "micro_task.h"


#include <iostream>
#include <iomanip>

using namespace iUtc;


int main(int argc, char**argv){
	bool printTime = false;
	int nStreams = 1;
	int nscale = 16;
	int blocksize = 1024;
	int memtype = pageable;
	int loop = 2;

	int nprocess = 1;
	int nthreads = 1;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"t:p:s:n:b:m:l:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
					  break;
			case 'p': nprocess = atoi(optarg);
					  break;
			case 's': nStreams = atoi(optarg);
					  break;
			case 'n': nscale = atoi(optarg);
					  break;
			case 'b': blocksize = atoi(optarg);
					  break;
			case 'm': memtype = atoi(optarg);
					  break;
			case 'l': loop = atoi(optarg);
					  break;
			case ':':
				std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
				break;
			case '?':
				std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
				break;
			default:
					  break;
		}
	}
	int procs = ctx.numProcs();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}

	int myproc = ctx.getProcRank();
	double runtime[4];

	Task<microTest<float>> microBench(ProcList(0), TaskType::gpu_task);
	/*
	 * test pageable mem
	 */
	microBench.init(nscale, blocksize, nStreams, loop, pageable);
	microBench.run(runtime);
	microBench.wait();
	if(myproc==0){
		int bytes = nscale*nscale*blocksize*nStreams*nStreams*sizeof(float);
		std::cout<<"pageable test results:"<<std::endl;
		std::cout<<"\t use default stream:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[0]<<std::endl;
		std::cout<<"\t\t kernel time(s):"<<runtime[1]<<std::endl;
		std::cout<<"\t\t data transfer bandwidth(GB/s):"<<(bytes*2*1e-9)/(runtime[0]-runtime[1])<<std::endl;
		std::cout<<"\t use multiple streams I:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[2]<<std::endl;
		std::cout<<"\t use multiple streams II:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[2]<<std::endl;
	}

	/*
	 * test pinned mem
	 */
	microBench.init(nscale, blocksize, nStreams, loop, pinmem);
	microBench.run(runtime);
	microBench.wait();
	if(myproc==0){
		int bytes = nscale*nscale*blocksize*nStreams*nStreams*sizeof(float);
		std::cout<<"pinned test results:"<<std::endl;
		std::cout<<"\t use default stream:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[0]<<std::endl;
		std::cout<<"\t\t kernel time(s):"<<runtime[1]<<std::endl;
		std::cout<<"\t\t data transfer bandwidth(GB/s):"<<(bytes*2*1e-9)/(runtime[0]-runtime[1])<<std::endl;
		std::cout<<"\t use multiple streams I:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[2]<<std::endl;
		std::cout<<"\t use multiple streams II:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[2]<<std::endl;
	}

	/*
	 * test unified mem
	 */
	microBench.init(nscale, blocksize, nStreams, loop, umem);
	microBench.run(runtime);
	microBench.wait();
	if(myproc==0){
		int bytes = nscale*nscale*blocksize*nStreams*nStreams*sizeof(float);
		std::cout<<"unified test results:"<<std::endl;
		std::cout<<"\t use default stream:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[0]<<std::endl;
		std::cout<<"\t use multiple streams I:"<<std::endl;
		std::cout<<"\t\t total time(s):"<<std::fixed<<std::setprecision(4)<<runtime[2]<<std::endl;

	}

}

