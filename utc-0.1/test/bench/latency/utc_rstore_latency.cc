/*
 * utc_rstore_latency.cc
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

/* main UTC namespace */
using namespace iUtc;

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<22)
#define MYBUFSIZE (MAX_MSG_SIZE + MESSAGE_ALIGNMENT)
#define SKIP_LARGE  10
#define LOOP_LARGE  100
#define SKIP_LLARGE 1
#define LOOP_LLARGE 10
#define LARGE_MESSAGE_SIZE  8192
#define LLARGE_MESSAGE_SIZE (8192*1024)


/*
 * user defined task implementation
 */
class RstoreWorker: public UserTaskBase{
private:
	GlobalScopedData<char> gdataBuff;
	int alligned_start_index;
	char *localdataBuff;
public:
	RstoreWorker()
	:gdataBuff(MYBUFSIZE),
	 localdataBuff(nullptr),
	 alligned_start_index(0){
	}

	void initImpl(){
		if(__localThreadId == 0){
			gdataBuff.init();
			localdataBuff = new char[MYBUFSIZE];
			localdataBuff = (char*)(((unsigned long)localdataBuff + (MESSAGE_ALIGNMENT-1))
					/MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT);
			alligned_start_index = (int)(((unsigned long)gdataBuff.getPtr() +
					(MESSAGE_ALIGNMENT-1))/MESSAGE_ALIGNMENT*MESSAGE_ALIGNMENT
					-(unsigned long)gdataBuff.getPtr());

			std::cout<<"Task init() method finish: \n"<<
					"\tthread_local_id: "<<__localThreadId<<std::endl<<
					"\tthread_global_id: "<<__globalThreadId<<std::endl<<
					"\tprocess_ingroup_id: "<<__processIdInGroup<<std::endl<<
					"\tprocess_inworld_id: "<<__processIdInWorld<<std::endl;
		}
		inter_Barrier();
	}
	void runImpl(){
		if(__globalThreadId==0){
			std::cout<<"Start rstore latency test:(us)"<<std::endl;
		}
		Timer timer;
		double latency = 0.0;
		inter_Barrier();
		for(int size = 1; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)){
			for(int i = 0; i<size; i++){
				gdataBuff[i+alligned_start_index] = (char)('0'+(__processIdInGroup + size)%10);
				localdataBuff[i] = (char)('0'+(__processIdInGroup + size)%10);
			}
			int loop = 10000;
			int skip = 1000;
			if(size > LARGE_MESSAGE_SIZE){
				loop = LOOP_LARGE;
				skip = SKIP_LARGE;
			} else if(size > LLARGE_MESSAGE_SIZE){
				loop = LOOP_LLARGE;
				skip = SKIP_LLARGE;
			}
			inter_Barrier();

			if(__processIdInGroup == 0){
				for(int i=0; i<loop + skip; i++){
					if(i==skip)
						timer.start();
					gdataBuff.rstoreblock(1, localdataBuff, alligned_start_index, size);
					gdataBuff.quiet();
				}
				latency = timer.stop()*1e6;
				latency /= loop;
			}
			if(__processIdInGroup == 1){
				//std::cout<<"global:"<<gdataBuff[alligned_start_index+size-1]<<" ";
				//std::cout<<"local:"<<gdataBuff[alligned_start_index+size-1]<<" ";
			}
			gdataBuff.barrier();
			if(__processIdInGroup == 1){
				//std::cout<<"local:"<<localdataBuff[size-1]<<std::endl;
			}
			if(__processIdInGroup == 0){
				std::cout<<size<<"\t\t"<<latency<<std::endl;
			}
		}

		sleep(1);
		inter_Barrier();
		if(__localThreadId == 0){
			std::cout<<"Task run method finish on thread "<<__globalThreadId<<std::endl;
			//delete localdataBuff;
			gdataBuff.destroy();
		}

	}

};


/*****************************************************
 * main() program
 ****************************************************/
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

	if(nprocs != ctx.numProcs() && nprocs != 2){
		std::cerr<<"process number not match with arguments '-p' and we need two processes!!!\n";
		return 1;
	}

	int process_ids[] = {0,1};
	ProcList plist(2, process_ids);
	Task<RstoreWorker> latencyTest(plist);

	std::cout<<"call task::init\n";
	latencyTest.init();

	std::cout<<"call task::run\n";
	latencyTest.run();
	latencyTest.wait();

	std::cout<<"call task::finish\n";
	latencyTest.finish();

	return 0;

}


