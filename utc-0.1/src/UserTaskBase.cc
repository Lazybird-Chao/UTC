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

void UserTaskBase::preInit(){
	__localThreadId = iUtc::getLrank();
	__globalThreadId = iUtc::getGrank();
	__processId = iUtc::getPrank();
	__numLocalThreads = iUtc::getLsize();
	__numGlobalThreads = iUtc::getGsize();
	__numProcesses = iUtc::getPsize();

	for(std::vector<iUtc::PrivateScopedDataBase *>::iterator item = __psDataRegistry.begin();
			item != __psDataRegistry.end(); item++){
		(*item)->init();
	}
}


void UserTaskBase::preExit(){
	for(auto& item: __psDataRegistry ){
		item->destroy();
	}
}

// end namespace iUtc


