/*
 * UserTaskBase.cc
 *
 *  Created on: Jun 17, 2016
 *      Author: chaoliu
 */
#include "UtcBasics.h"
#include "UserTaskBase.h"
#include "TaskBase.h"
#include "TaskUtilities.h"

#include <iostream>


thread_local int UserTaskBase::__localThreadId = -1;
thread_local int UserTaskBase::__globalThreadId = -1;
thread_local int UserTaskBase::__processIdInWorld = -1;
thread_local int UserTaskBase::__processIdInGroup = -1;

#if ENABLE_GPU_TASK
thread_local cudaStream_t UserTaskBase::__streamId = nullptr;
thread_local int UserTaskBase::__deviceId = -1;
#endif
UserTaskBase::UserTaskBase(){

#if ENABLE_SCOPED_DATA
	__psDataRegistry.clear();
#endif

}

UserTaskBase::~UserTaskBase(){
#if ENABLE_SCOPED_DATA
	__psDataRegistry.clear();
#endif

}


// usually initialize data members and prepare for computation,
// you should implement this method and can overload with arguments
// in derived class
void UserTaskBase::initImpl(){
	std::cerr<<"Using the template User task class!\n";
}



// usually implement the main algorithm, doing computation and communication,
// you should implement this method in derived class
void UserTaskBase::runImpl(){
	std::cerr<<"Using the template User task class!\n";
}




#if ENABLE_SCOPED_DATA
void UserTaskBase::registerPrivateScopedData(iUtc::PrivateScopedDataBase *psData){
	__psDataRegistry.push_back(psData);
}
#endif

void UserTaskBase::preInit(int lrank,
							int trank,
							int prank,
							int numLocalThreads,
							int numProcesses,
							int numTotalProcesses,
							int numTotalThreads,
							std::map<int,int> *worldRankTranslate,
							std::map<int,int> *groupRankTranslate,
							void* gpuCtx){
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"pre Taskinit start"<<std::endl;
#endif
	__localThreadId = lrank;
	__globalThreadId = trank;
	__processIdInWorld = prank;
	__processIdInGroup = worldRankTranslate->at(prank);
	__numLocalThreads = numLocalThreads;
	__numGlobalThreads = numTotalThreads;
	__numWorldProcesses = numTotalProcesses;
	__numGroupProcesses = numProcesses;
	__worldRankTranslate = worldRankTranslate;
	__groupRankTranslate = groupRankTranslate;

	__fastIntraSync.init(numLocalThreads);

#if ENABLE_GPU_TASK
	iUtc::UtcGpuContext *gCtx = (iUtc::UtcGpuContext*)gpuCtx;
	if(gCtx){
		__streamId = gCtx->getBoundStream();
		__deviceId = gCtx->getCudaDeviceId();
	}
#endif

#if ENABLE_SCOPED_DATA
	if(__psDataRegistry.size()>0){
		for(std::vector<iUtc::PrivateScopedDataBase *>::iterator item = __psDataRegistry.begin();
				item != __psDataRegistry.end(); item++){
			(*item)->init();
		}
	}

#ifdef USE_INTERNALSHMEM
	iUtc::getCurrentTask()->getTaskMpiWindow()->scoped_win_init();
#endif

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"pre Taskinit finish"<<std::endl;
#endif

#endif

}


void UserTaskBase::preExit(){
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"pre Taskexit start"<<std::endl;
#endif
#if ENABLE_SCOPED_DATA
	for(auto& item: __psDataRegistry ){
		item->destroy();
	}

#ifdef USE_INTERNALSHMEM
	iUtc::getCurrentTask()->getTaskMpiWindow()->scoped_win_finalize();
#endif

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"pre Taskexit finish"<<std::endl;
#endif

#endif

}

// end namespace iUtc


