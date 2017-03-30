/*
 * hc_main.cc
 *
 *  Created on: Mar 23, 2017
 *      Author: Chao
 */

#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "sgpu/hc_task_sgpu.h"
#include "task.h"
#include "typeconfig.h"

#define H 1.0

int main(int argc, char **argv){
	int WIDTH = 400;
	int HEIGHT = 600;
	FTYPE EPSILON = 0.1;
	bool output = false;

	bool printTime = false;
	int blockSize = 16;
	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int opt;
	extern char* optarg;
	extern int optind, optopt;
	opt=getopt(argc, argv, "vt:p:m:h:w:e:b:o");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 'm': mtype = atoi(optarg);
			  break;
		case 'w':
			WIDTH = atoi(optarg);
			break;
		case 'h':
			HEIGHT = atoi(optarg);
			break;
		case 'e':
			EPSILON = atof(optarg);
			break;
		case 'b':
			blockSize = atoi(optarg);
			break;
		case 'o':
			output = true;
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
		opt=getopt(argc, argv, "vt:p:m:h:w:e:b:o");
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

	if(WIDTH<=0 || HEIGHT<=0){
		std::cerr<<"illegal width or height"<<std::endl;
		exit(1);
	}


	/*
	 *
	 */
	double runtime[5];
	int iters;
	FTYPE *domainMatrix = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	Task<hcSGPU> hc(ProcList(0), TaskType::gpu_task);
	hc.init(WIDTH, HEIGHT, EPSILON, domainMatrix);
	hc.run(runtime, &iters, blockSize, memtype);
	hc.wait();


	/*
	 * output
	 */
	if(output){
		char filename[30] = "output.txt";
		Task<Output<FTYPE>> fout(ProcList(0));
		fout.run(domainMatrix, WIDTH, HEIGHT, filename);
	}

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tDomain size: "<<WIDTH<<" X "<<HEIGHT<<std::endl;
		std::cout<<"\tAccuracy: "<<EPSILON<<std::endl;
		std::cout<<"\tIterations: "<<iters<<std::endl;
		std::cout<<"\tMemtype: "<<mtype<<std::endl;
		std::cout<<"\tTime info: "<<std::endl;
		std::cout<<"\t\ttotal time: "<<runtime[0]<<"(s)"<<std::endl;
		std::cout<<"\t\tkernel run time: "<<runtime[1]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy in time: "<<runtime[2]<<"(s)"<<std::endl;
		std::cout<<"\t\tmemcpy out time: "<<runtime[3]<<"(s)"<<std::endl;
		//std::cout<<"\t\thost compute time: "<<runtime[0]<<"(s)"<<std::endl;

	}

	for(int i=0; i<5; i++)
		runtime[i] *= 1000;
	print_time(5, runtime);

	return 0;

}







