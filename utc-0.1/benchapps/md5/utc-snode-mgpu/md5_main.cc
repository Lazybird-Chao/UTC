/*
 * md5_main.cc
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "md5.h"
#include "task.h"
#include "mgpu/md5_task_mgpu.h"


int main(int argc, char* argv[]){
	bool printTime = false;
	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;

	config_t configArgs;
	configArgs.input_set = 0;
	configArgs.iterations = 1;
	configArgs.outflag = 0;
	int blocksize = 256;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vt:p:m:i:c:ob:");
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
		case 'i':
			configArgs.input_set = atoi(optarg);
			break;
		case 'c':
			configArgs.iterations = atoi(optarg);
			break;
		case 'o':
			configArgs.outflag = 1;
			break;
		case 'b':
			blocksize = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vt:p:m:i:c:ob:");
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;


	/*
	 * preparing data
	 */
	std::cout<<"Generating random input data set ..."<<std::endl;
	Task<RandomInput> dataGen(ProcList(0));
	dataGen.run(&configArgs);
	dataGen.wait();

	/*
	 * do md5
	 */
	std::cout<<"Start MD5 processing ..."<<std::endl;
	double runtime_m[8][4];
	Task<MD5MGPU> md5(ProcList(nthreads, 0), TaskType::gpu_task);
	md5.init(&configArgs);
	md5.run(runtime_m, blocksize, memtype);
	md5.wait();
	double runtime[4]={0,0,0,0};
	for(int i=0; i<nthreads; i++)
		for(int j=0; j<4; j++)
			runtime[j]+= runtime_m[i][j];
	for(int j=0; j<4; j++)
		runtime[j] /= nthreads;


	/*
	 * output to file
	 */
	if(configArgs.outflag){
		Task<Output> fileOut(ProcList(0));
		fileOut.run(&configArgs);
		fileOut.wait();
	}

	if(configArgs.inputs)
		free(configArgs.inputs);
	if(configArgs.out)
		free(configArgs.out);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tprocess data info:"<<std::endl;
		std::cout<<"\t\tnumber buffs:"<<datasets[configArgs.input_set].numbufs<<std::endl;
		std::cout<<"\t\tbuff size(Bytes):"<<datasets[configArgs.input_set].bufsize<<std::endl;
		std::cout<<"\t\tMemtype: "<<mtype<<std::endl;
		std::cout<<"\ttime info:"<<std::endl;
		std::cout<<"\t\ttotal time: "<<std::fixed<<std::setprecision(4)<<1000*(runtime[0])<<"(ms)"<<std::endl;
		std::cout<<"\t\tkernel time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopy in: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopy out: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;

	}

	for(int i=0; i<4; i++)
		runtime[i] *= 1000;
	print_time(4, runtime);

	return 0;

}

