/*
 * md5_main.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../../common/helper_getopt.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"

#include "md5.h"
#include "task.h"

#define MAX_THREADS 64
using namespace iUtc;

int main(int argc, char* argv[]){
	bool printTime = false;
	int nthreads=1;
	int nprocess=1;

	config_t configArgs;
	configArgs.input_set = 0;
	configArgs.iterations = 1;
	configArgs.outflag = 0;
	char *inputFile = nullptr;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);
	std::cout<<"UTC context initialized !\n";


	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "vt:p:i:c:f:o");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 'i':
			configArgs.input_set = atoi(optarg);
			break;
		case 'f':
			inputFile = optarg;
			break;
		case 'c':
			configArgs.iterations = atoi(optarg);
			break;
		case 'o':
			configArgs.outflag = 1;
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vt:p:i:c:f:o");
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs || nprocess > 1){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	/*if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
		return 1;
	}*/

	/*
	 * preparing data
	 */
	std::cout<<"Generating random input data set ..."<<std::endl;
	Task<RandomInput> dataGen(ProcList(0));
	dataGen.run(&configArgs, inputFile, true);
	//dataGen.run(&configArgs, nullptr, false);
	dataGen.wait();

	//
	//toFile((char*)configArgs.inputs, configArgs.numinputs, configArgs.size, inputFile, true);
	//std::cout<<configArgs.numinputs<<" "<<configArgs.size<<std::endl;
	increaseBy(4, &configArgs);


	/*
	 * do md5
	 */
	std::cout<<"Start MD5 processing ..."<<std::endl;
	double runtime_m[MAX_THREADS][1];
	Task<MD5Worker> md5(ProcList(nthreads, 0));
	md5.init(&configArgs);
	md5.run(runtime_m);
	md5.wait();

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

	double runtime = 0.0;
	for(int i=0; i<nthreads; i++)
		runtime += runtime_m[i][0];
	runtime /= nthreads;
	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tprocess data info:"<<std::endl;
		std::cout<<"\t\tnumber buffs:"<<configArgs.numinputs<<std::endl;
		std::cout<<"\t\tbuff size(Bytes):"<<configArgs.size<<std::endl;
		std::cout<<"\ttime info:"<<std::endl;
		std::cout<<"\t\ttotal time: "<<std::fixed<<std::setprecision(4)<<1000*(runtime)<<"(ms)"<<std::endl;
	}

	runtime *= 1000;
	print_time(1, &runtime);

	return 0;

}


