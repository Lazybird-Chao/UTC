/*
 * test_gsData.cc
 *
 *  Created on: Jun 27, 2016
 *      Author: chaoliu
 */
#include "Utc.h"
#include "../bench/helper_getopt.h"

#include <iostream>

using namespace iUtc;

class TaskGsData: public UserTaskBase{
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
			//////////////////////////////for array data
			sleep(__globalThreadId);
			if(__globalThreadId ==0){
				std::cout<<"store value to global data objs in thread "
						<<__globalThreadId<<std::endl;
				for(int i=0; i< arrayVar.getSize(); i++)
					arrayVar.store(__globalThreadId*arrayVar.getSize()+i, i);
			}
			inter_Barrier();

			sleep(__numGlobalThreads-__globalThreadId);
			if(__globalThreadId == 1){
				std::cout<<"load value from global data objs in thread "
						<<__globalThreadId<<std::endl;
				std::cout<<"\t"<<arrayVar.load(10)<<std::endl;
			}
			inter_Barrier();

			sleep(__numGlobalThreads-__globalThreadId);
			if(__globalThreadId == 1){
				std::cout<<"remote load value from global data objs in thread "
						<<__globalThreadId<<std::endl;
				std::cout<<"\t"<<arrayVar.rload((__processId+1)%__numProcesses,10)<<std::endl;
			}
			inter_Barrier();

			sleep(__globalThreadId);
			if(__globalThreadId ==1){
				std::cout<<"remote store value to global data objs in thread "
						<<__globalThreadId<<std::endl;
				for(int i=0; i< arrayVar.getSize(); i++)
					arrayVar.rstore((__processId+1)%__numProcesses,__globalThreadId*arrayVar.getSize()+i, i);
				arrayVar.rstoreFence();
			}
			inter_Barrier();

			sleep(__globalThreadId);
			if(__globalThreadId ==0){
				std::cout<<"load after remote store of global data objs in thread "
						<<__globalThreadId<<std::endl;
				for(int i=0; i< arrayVar.getSize(); i++)
					std::cout<<"\t"<<arrayVar.load(i)<<std::endl;
			}
			inter_Barrier();



		}

	private:
		GlobalScopedData<double> arrayVar;

	public:
		TaskGsData()
		:arrayVar(this, 20){

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
	std::cout<<"here "<<nproc<<" "<<myproc<<std::endl;
	ProcList plist;
	for(int i=0; i<nprocs; i++)
		for(int j=0; j<nthreads; j++)
			plist.push_back(i);

	Task<TaskGsData>  test(plist);

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



