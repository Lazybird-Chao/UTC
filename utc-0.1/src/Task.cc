#include "Task.h"
#include "RootTask.h"
#include <functional>

namespace iUtc{

std::ofstream* getProcOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid != 0)
	{
		// this function must be called in main thread, not in some task's thread
		// main thread in a process is actually root, root task id is 0
		std::cout<<"Error, getProcOstream() must be called in main thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	RootTask *root = TaskManager::getRootTask();
	std::ofstream *procOstream = root->getProcOstream();
	if(!procOstream)
	{
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
		std::cout<<"Error, getThreadOstream() must be called in task's thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	ThreadPrivateData *tpd = TaskBase::getThreadPrivateData();
	if(!tpd->threadOstream)
	{
		std::string filename = "./log/";
		filename.append((TaskManager::getCurrentTask())->getName());
		filename.append("-thread");
		filename.append(std::to_string(TaskManager::getCurrentThreadRankinTask()));
		filename.append(".log");
		tpd->threadOstream = new std::ofstream(filename);
	}

	//return std::ref(*(tpd->threadOstream));
	return tpd->threadOstream;

}

int getTid()
{
	static thread_local int taskid = -1;
	if(taskid == -1)
	{
		taskid = TaskManager::getCurrentTaskId();
	}
	return taskid;
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


}// namespcae iUtc
















