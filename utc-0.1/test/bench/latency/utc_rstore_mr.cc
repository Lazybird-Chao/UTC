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

/* main UTC namespace */
using namespace iUtc;


#define ITERS_SMALL     (500)
#define ITERS_LARGE     (50)
#define LARGE_THRESHOLD (8192)
#define MAX_MSG_SZ (1<<22)
#define LOOP (10)

#define MESSAGE_ALIGNMENT (1<<12)
#define MYBUFSIZE (MAX_MSG_SZ * ITERS_LARGE + MESSAGE_ALIGNMENT)

/*
 * user defined task implementation
 */
class RstoreWorker: public UserTaskBase{
private:
	GlobalScopedData<char> gdataBuff;
	int alligned_start_index;
	char *localdataBuff;

public:
	RstoreWorker():
		gdataBuff(MYBUFSIZE),
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
			/*
			std::cout<<"Task init() method finish: \n"<<
								"\tthread_local_id: "<<__localThreadId<<std::endl<<
								"\tthread_global_id: "<<__globalThreadId<<std::endl<<
								"\tprocess_ingroup_id: "<<__processIdInGroup<<std::endl<<
								"\tprocess_inworld_id: "<<__processIdInWorld<<std::endl;
								*/
		}
		inter_Barrier();
	}

	void runImpl(double msgRate[23], bool showtime){
		if(__globalThreadId==0){
			//std::cout<<"Start rstore message rate test:(us)"<<std::endl;
		}
		Timer timer;

		for(int i = 0; i<MYBUFSIZE; i++){
			gdataBuff[i] = (char)('0'+(__processIdInGroup + MYBUFSIZE)%10);
			localdataBuff[i] = (char)('0'+(__processIdInGroup + MYBUFSIZE)%10);
		}
		inter_Barrier();

		/*
		 * warm up
		 */
		if(__processIdInGroup == 0){
			for(int i =0; i<(ITERS_LARGE*MAX_MSG_SZ); i+= MAX_MSG_SZ){
					gdataBuff.rstoreblock((__processIdInGroup+1)%__numGroupProcesses,  //pe
							localdataBuff + i,  //address base
							alligned_start_index + i,  //start offset
							MAX_MSG_SZ);		//data count
			}
		}
		gdataBuff.barrier();

		int count = 0;
		for(int size = 1; size <= MAX_MSG_SZ; size <<= 1){
			int iters = size < LARGE_THRESHOLD ? ITERS_SMALL : ITERS_LARGE;
			if(__processIdInGroup == 0){
				timer.start();
				for(int j = 0; j<LOOP; j++){
					for(int i = 0, offset = 0; i<iters; i++, offset+=size){
						gdataBuff.rstoreblock((__processIdInGroup+1)%__numGroupProcesses,
								localdataBuff + offset,
								alligned_start_index + offset,
								size);
					}
					gdataBuff.quiet();
				}
				double time = timer.stop();
				msgRate[count] = (double)iters/time*size*LOOP/1e6;
				if(showtime){
					std::cout<<size<<"\t\t"<<msgRate[count]<<std::endl;
				}
				count++;
			}
		}

		sleep(1);
		inter_Barrier();
		if(__localThreadId == 0){
			gdataBuff.destroy();
			std::cout<<"Task run method finish on thread "<<__globalThreadId<<std::endl;
		}

	}

};



/*****************************************************
 * main() program
 ****************************************************/
int main(int argc, char* argv[]){

	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads = 1;
	int nprocs = 1;
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
	double msgrate_m[32][23];
	std::vector<Task<RstoreWorker>*> msgRateTaskList;
	for(int i = 0; i < ntasks; i++){
		msgRateTaskList.push_back(new Task<RstoreWorker>(plist, TaskType::cpu_task, 512*1024*1024));
	}

	std::cout<<"call task::init\n";
	for(int i = 0; i < ntasks; i++)
		msgRateTaskList[i]->init();

	std::cout<<"call task::run\n";
	for(int i = 0; i<ntasks; i++){
		if(i == 0)
			msgRateTaskList[i]->run(msgrate_m[i], true);
		else msgRateTaskList[i]->run(msgrate_m[i], false);
	}
	for(int i = 0; i<ntasks; i++)
		msgRateTaskList[i]->wait();

	std::cout<<"call task::finish\n";
	for(int i = 0; i<ntasks; i++)
		msgRateTaskList[i]->finish();

	ctx.Barrier();
	if(ctx.getProcRank() == 0){
		std::cout<<"average message: \n";
		double avg_mr[23];
		for(int i = 0; i<23; i++){
			avg_mr[i] = 0;
			for(int j = 0; j<ntasks; j++)
				avg_mr[i] += msgrate_m[j][i];
			//avg_mr[i] /= ntasks;
			std::cout<<(1<<i)<<"\t\t"<<avg_mr[i]<<std::endl;
		}
	}

	return 0;

}



