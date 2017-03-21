/*
 * UserTaskBase.cc
 *
 *  Created on: Jun 17, 2016
 *      Author: chaoliu
 */

#include "UserTaskBase.h"
#include "TaskUtilities.h"

#include <iostream>


thread_local int UserTaskBase::__localThreadId = -1;
thread_local int UserTaskBase::__globalThreadId = -1;
thread_local int UserTaskBase::__processId = -1;

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
							int numTotalThreads,
							std::map<int,int> *worldRankTranslate,
							void* gpuCtx){
	__localThreadId = lrank;
	__globalThreadId = trank;
	__processId = prank;
	__numLocalThreads = numLocalThreads;
	__numGlobalThreads = numTotalThreads;
	__numProcesses = numProcesses;
	__worldRankTranslate = worldRankTranslate;

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
#endif

}


void UserTaskBase::preExit(){
#if ENABLE_SCOPED_DATA
	for(auto& item: __psDataRegistry ){
		item->destroy();
	}
#endif

}

// end namespace iUtc


