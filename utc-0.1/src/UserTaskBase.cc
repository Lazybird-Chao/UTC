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
UserTaskBase::UserTaskBase(){

	__psDataRegistry.clear();
}

UserTaskBase::~UserTaskBase(){
	__psDataRegistry.clear();
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


void UserTaskBase::registerPrivateScopedData(iUtc::PrivateScopedDataBase *psData){
	__psDataRegistry.push_back(psData);
}

void UserTaskBase::preInit(int lrank,
							int trank,
							int prank,
							int numLocalThreads,
							int numProcesses,
							int numTotalThreads){
	__localThreadId = lrank;
	__globalThreadId = trank;
	__processId = prank;
	__numLocalThreads = numLocalThreads;
	__numGlobalThreads = numTotalThreads;
	__numProcesses = numProcesses;

	if(__psDataRegistry.size()>0){
		for(std::vector<iUtc::PrivateScopedDataBase *>::iterator item = __psDataRegistry.begin();
				item != __psDataRegistry.end(); item++){
			(*item)->init();
		}
	}
}


void UserTaskBase::preExit(){
	for(auto& item: __psDataRegistry ){
		item->destroy();
	}
}

// end namespace iUtc


