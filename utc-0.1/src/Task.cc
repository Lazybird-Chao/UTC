#include "Task.h"
#include "RootTask.h"
#include "../include/TaskUtilities.h"
#include "UniqueExeTag.h"

#include <functional>
#include "boost/filesystem.hpp"

namespace iUtc{


/* Task utility methods */

std::ofstream* getProcOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid != 0)
	{
		// this function must be called in main thread, not in some task's thread
		// main thread in a process is actually root, root task id is 0
		std::cerr<<"Error, getProcOstream() must be called in main thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	RootTask *root = TaskManager::getRootTask();
	std::ofstream *procOstream = root->getProcOstream();
	if(!procOstream)
	{
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/Proc";
		filename.append(std::to_string(root->getCurrentProcRank()));
		filename.append(".log");
		procOstream = new std::ofstream(filename);
		root->setProcOstream(*procOstream);

	}
	//return std::ref(*procOstream);
	return procOstream;
}

std::ofstream* getThreadOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid == 0)
	{
		std::cerr<<"Error, getThreadOstream() must be called in task's thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	ThreadPrivateData *tpd = TaskBase::getThreadPrivateData();
	if(!tpd->threadOstream)
	{
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/";
		filename.append((TaskManager::getCurrentTask())->getName());
		filename.append("-");
		filename.append(std::to_string(TaskManager::getCurrentTaskId()));
		filename.append("-thread");
		filename.append(std::to_string(TaskManager::getCurrentThreadRankinTask()));
		filename.append(".log");
		tpd->threadOstream = new std::ofstream(filename);
	}

	//return std::ref(*(tpd->threadOstream));
	return tpd->threadOstream;

}

int getTaskId()
{
	static thread_local int taskid = -1;
	if(taskid == -1)
	{
		taskid = TaskManager::getCurrentTaskId();
	}
	return taskid;
}

int getParentTaskId()
{
	static thread_local int parentTid = -1;
	if(parentTid == -1)
	{
		parentTid = TaskManager::getParentTaskId();
	}
	return parentTid;
}

int getTrank()
{
	static thread_local int threadrank=-1;
	if(threadrank ==-1)
	{
		threadrank = TaskManager::getCurrentThreadRankinTask();
	}
	return threadrank;
}

int getPrank()
{
	static thread_local int procrank = -1;
	if(procrank == -1)
	{
		procrank = TaskManager::getCurrentProcessRankinTask();
	}
	return procrank;
}

int getLrank()
{
	static thread_local int localrank = -1;
	if(localrank == -1)
	{
		localrank = TaskManager::getCurrentThreadRankInLocal();
	}
	return localrank;
}


int getLsize()
{
	static thread_local int localthreads = -1;
	if(localthreads ==-1)
	{
		localthreads = TaskManager::getCurrentTask()->getNumLocalThreads();
	}
	return localthreads;
}

int getGsize()
{
	static thread_local int globalthreads = -1;
	if(globalthreads == -1)
	{
		globalthreads = TaskManager::getCurrentTask()->getNumTotalThreads();
	}
	return globalthreads;
}


TaskBase* getCurrentTask()
{
	static thread_local TaskBase* currentTaskPtr= nullptr;
	if(!currentTaskPtr)
	{
		currentTaskPtr = TaskManager::getCurrentTask();
	}
	return currentTaskPtr;
}

TaskBase* getParentTask()
{
	static thread_local TaskBase* ParentTaskPtr= nullptr;
	if(!ParentTaskPtr)
	{
		ParentTaskPtr = TaskManager::getParentTask();
	}
	return ParentTaskPtr;
}



bool getUniqueExecution(){
	static thread_local UniqueExeTag *uniqueExeTagPtr = nullptr;
	static thread_local int mylocalrank = -1;
	if(uniqueExeTagPtr== nullptr){
		uniqueExeTagPtr = TaskBase::getThreadPrivateData()->taskUniqueExeTagObj;
		mylocalrank = TaskManager::getCurrentThreadRankInLocal();
	}
	return uniqueExeTagPtr->getUniqueExe(mylocalrank);
}


}// namespcae iUtc
















