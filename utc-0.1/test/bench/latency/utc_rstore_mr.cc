/*
 * utc_rstore_mr.cc
 *
 *  Created on: Sep 28, 2017
 *      Author: chaoliu
 */

/* main UTC header file */
#include "Utc.h"

/* other standard header file */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>

#define ITERS_SMALL     (500)
#define ITERS_LARGE     (50)
#define LARGE_THRESHOLD (8192)
#define MAX_MSG_SZ (1<<22)

#define MESSAGE_ALIGNMENT (1<<12)
#define MYBUFSIZE (MAX_MSG_SZ * ITERS_LARGE + MESSAGE_ALIGNMENT)

#define SHARE_MEM_SIZE ()



/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char* argv[]){

	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads = 1;
	int nprocs;
	int ntasks = 1;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:n:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case 'n':
			ntasks = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:n:");
	}

	if(nprocs != ctx.numProcs() && nprocs != 2){
		std::cerr<<"process number not match with arguments '-p' and we need two processes!!!\n";
		return 1;
	}

	int process_ids[] = {0,1};
	ProcList plist(2, process_ids);
	std::vector<Task<RstoreWorker>*> msgRateTaskList;
	for(int i = 0; i < ntasks; i++){
		msgRateTaskList.push_back(new Task<RstoreWorker>(plist));
	}

	std::cout<<"call task::init\n";
	for(int i = 0; i < ntasks; i++)
		msgRateTaskList[i]->init();

	std::cout<<"call task::run\n";
	for(int i = 0; i<ntasks; i++)
		msgRateTaskList[i]->run();
	for(int i = 0; i<ntasks; i++)
		msgRateTaskList[i]->wait();

	std::cout<<"call task::finish\n";
	for(int i = 0; i<ntasks; i++)
		msgRateTaskList[i]->finish();

	return 0;

}



