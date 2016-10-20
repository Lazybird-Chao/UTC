/*
 * test_psData.cc
 *
 *  Created on: Jun 20, 2016
 *      Author: chaoliu
 */
#include "Utc.h"
#include "../bench/helper_getopt.h"

#include <iostream>


using namespace iUtc;

class TaskPsData: public UserTaskBase{
public:
	void initImpl(){
		sleep(__globalThreadId);
		std::cout<<"This is task init() method. \n"
				<<"gId: "<<__globalThreadId<<std::endl
				<<"lId: "<<__localThreadId<<std::endl
				<<"gSize: "<<__numGlobalThreads<<std::endl
				<<"lSize: "<<__numLocalThreads<<std::endl;
	}

	void runImpl(){
		sleep(__globalThreadId);
		std::cout<<"store value to private data objs in thread "
				<<__globalThreadId<<std::endl;
		for(int i=0; i< arrayVar.getSize(); i++)
			arrayVar.store(__globalThreadId*arrayVar.getSize()+i, i);
		inter_Barrier();
		sleep(__numGlobalThreads-__globalThreadId);
		std::cout<<"load value from private data objs in thread "
				<<__globalThreadId<<std::endl;
		std::cout<<"\t"<<arrayVar.load(10)<<std::endl;
		inter_Barrier();

		sleep(__globalThreadId);
		std::cout<<"store value to private data objs in thread "
				<<__globalThreadId<<std::endl;
		singleVar = __globalThreadId;
		inter_Barrier();
		sleep(__numGlobalThreads-__globalThreadId);
		std::cout<<"load value from private data objs in thread "
				<<__globalThreadId<<std::endl;
		int var = singleVar;
		std::cout<<"\t"<<var<<std::endl;



	}

private:
	PrivateScopedData<int> singleVar;
	PrivateScopedData<double> arrayVar;

public:
	TaskPsData()
	:singleVar(this),
	 arrayVar(this, 20){

	}

};


int main(int argc, char* argv[]){

	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads;
	int nprocs;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:");
	}

	int nproc = ctx.numProcs();
	int myproc = ctx.getProcRank();

	ProcList plist;
	for(int i=0; i<nprocs; i++)
		for(int j=0; j<nthreads; j++)
			plist.push_back(i);

	Task<TaskPsData>  test(plist);

	std::cout<<"call task::init\n";
	test.init();
	test.wait();

	std::cout<<"call task::run\n";
	test.run();
	test.wait();

	std::cout<<"call task::finish\n";
	test.finish();

	return 0;
}
