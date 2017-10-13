/*
 * hc_main.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */
#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "../../common/helper_err.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "./gpu/hc_task.h"
#include "typeconfig.h"

#define H 1.0
#define MAX_THREADS 64
#define MAX_TIMER 9

int main(int argc, char **argv){
	int WIDTH = 400;
	int HEIGHT = 600;
	FTYPE EPSILON = 0.1;
	bool output = false;

	bool printTime = false;
	int nthreads=1;
	int nprocess=1;
	int blockSize = 16;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	if(ctx.getProcRank()==0)
		std::cout<<"UTC context initialized !\n";

	int opt;
	extern char* optarg;
	extern int optind, optopt;
	opt=getopt(argc, argv, "vt:p:h:w:e:b:o");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
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
		opt=getopt(argc, argv, "vt:p:h:w:e:b:o");
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	if(nthreads != 1){
		std::cerr<<"Only one thread for a task on each node !!!\n";
		return 1;
	}
	if(WIDTH<=0 || HEIGHT<=0){
		std::cerr<<"illegal width or height"<<std::endl;
		exit(1);
	}

	/*
	 *
	 */
	double runtime_m[MAX_THREADS][MAX_TIMER];
	int iters;
	FTYPE *domainMatrix;
	if(myproc == 0)
		domainMatrix = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	ProcList plist;
	for(int i = 0; i<nprocess; i++)
		for(int j = 0; j<nthreads; j++)
			plist.push_back(i);
	Task<HeatConductionWorker> hc(plist, TaskType::gpu_task);
	hc.init(WIDTH, HEIGHT, EPSILON, domainMatrix);
	hc.run(runtime_m, &iters, blockSize);
	hc.wait();



	/*
	 * output
	 */
	if(output){
		char filename[30] = "output.txt";
		Task<Output<FTYPE>> fout(ProcList(0));
		fout.run(domainMatrix, WIDTH, HEIGHT, filename);
	}

	if(myproc == 0){
		double runtime[MAX_TIMER]={0,0,0,0,0,0,0,0,0};
		for(int i=0; i<nthreads; i++)
			for(int j=0; j<MAX_TIMER; j++)
				runtime[j]+= runtime_m[i][j];
		for(int j=0; j<MAX_TIMER; j++)
			runtime[j] /= nthreads;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tDomain size: "<<WIDTH<<" X "<<HEIGHT<<std::endl;
			std::cout<<"\tAccuracy: "<<EPSILON<<std::endl;
			std::cout<<"\tIterations: "<<iters<<std::endl;
			std::cout<<"\tTime info: "<<std::endl;
			std::cout<<"\t\ttotal time: "<<runtime[0]<<"(s)"<<std::endl;
			std::cout<<"\t\tcompute time: "<<runtime[1]<<"(s)"<<std::endl;
			std::cout<<"\t\tcomm time: "<<runtime[2]<<"(s)"<<std::endl;
			std::cout<<"\t\tcopy time: "<<runtime[3]<<"(s)"<<std::endl;

		}
		for(int i=0; i<MAX_TIMER; i++)
			runtime[i] *= 1000;
		print_time(4, runtime);
	}
	ctx.Barrier();
	return 0;

}
