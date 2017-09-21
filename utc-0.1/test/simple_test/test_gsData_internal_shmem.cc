/*
 * test_gsData_internal_shmem.cc
 *
 *  Created on: Sep 14, 2017
 *      Author: chaoliu
 */

#include "Utc.h"
#include "../../../benchapps/common/helper_getopt.h"
#include "../../../benchapps/common/helper_err.h"
#include "../../../benchapps/common/helper_printtime.h"

#include <iostream>
#include <string>

using namespace iUtc;

class TaskGsData: public UserTaskBase{
private:
	GlobalScopedData<double> arrayVar;
public:
	TaskGsData()
	:arrayVar(100){

	}
	void initImpl(){
		sleep(__globalThreadId);
		arrayVar.init();
		std::cout<<"This is task init() method. \n"
						<<"gId: "<<__globalThreadId<<std::endl
						<<"lId: "<<__localThreadId<<std::endl
						<<"gSize: "<<__numGlobalThreads<<std::endl
						<<"lSize: "<<__numLocalThreads<<std::endl
						<<"prankInGroup: "<<__processIdInGroup<<std::endl
						<<"prankInWorld: "<<__processIdInWorld<<std::endl;
	}

	void runImpl(){
		sleep(__globalThreadId);
		if(__processIdInGroup ==0){
			std::cout<<"store value to global data objs in thread "
					<<__globalThreadId<<std::endl;
			for(int i=0; i< arrayVar.getSize(); i++){
				arrayVar.store(__processIdInWorld*arrayVar.getSize()+i, i);
				std::cout<<i<<" ";
			}
			std::cout<<std::endl;
		}
		inter_Barrier();


		sleep(__numGlobalThreads-__globalThreadId);
		if(__processIdInGroup == 1){
			std::cout<<"load value from global data objs in thread "
					<<__globalThreadId<<std::endl;
			std::cout<<"\t"<<arrayVar.load(10)<<std::endl;
		}
		inter_Barrier();


		sleep(__numGlobalThreads-__globalThreadId);
		if(__processIdInGroup == 1){
			std::cout<<"remote load value from global data objs in thread "
					<<__globalThreadId<<std::endl;
			std::cout<<"\t"<<arrayVar.rload(0,10)<<std::endl;
		}
		inter_Barrier();

		sleep(__globalThreadId);
		if(__processIdInGroup ==1){
			std::cout<<"remote store value to global data objs in thread "
					<<__globalThreadId<<std::endl;
			for(int i=0; i< arrayVar.getSize(); i++)
				arrayVar.rstore(0,__processIdInWorld*arrayVar.getSize()+i, i);
			arrayVar.fence();
		}
		inter_Barrier();

		sleep(__globalThreadId);
		if(__processIdInGroup ==0){
			std::cout<<"load after remote store of global data objs in thread "
					<<__globalThreadId<<std::endl;
			for(int i=0; i< arrayVar.getSize(); i++){
				std::cout<<" "<<arrayVar.load(i)<<" ";
			}
			std::cout<<std::endl;
		}
		inter_Barrier();
	}

};



int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads = 1;
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

	if(nprocs != ctx.numProcs()){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	std::cout<<getpid()<<"---->"<<ctx.getProcRank()<<std::endl;

	int process_ids[] = {0,1,2,3,4};
	ProcList plist(2, process_ids);
	Task<TaskGsData>  test1(plist);
	ProcList plist2(2, &process_ids[0]);
	Task<TaskGsData>  test2(plist2);

	std::cout<<"call task::init\n";
	test1.init();
	test1.wait();

	test2.init();
	test2.wait();

	std::cout<<"call task::run\n";
	test1.run();
	test1.wait();

	test2.run();
	test2.wait();

	std::cout<<"call task::finish\n";
	test1.finish();
	test2.finish();

	//char c;
	//std::cin>>c;

	return 0;
}

