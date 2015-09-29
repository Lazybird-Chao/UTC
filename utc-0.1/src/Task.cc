#include "Task.h"
#include "RootTask.h"
#include <functional>

namespace iUtc{

std::ofstream& getProcOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid != 0)
	{
		// this function must be called in main thread, not in some task's thread
		// main thread in a process is actually root, root task id is 0
		std::cout<<"Error, getProcOstream() must be called in main thread\n";
		std::ofstream output;
		return std::ref(output);
	}
	RootTask *root = TaskManager::getRootTask();
	std::ofstream *procOstream = root->getProcOstream();
	if(!procOstream)
	{
		std::string filename = "Proc";
		filename.append(std::to_string(root->getCurrentProcRank()));
		filename.append(".log");
		procOstream = new std::ofstream(filename);
		root->setProcOstream(*procOstream);

	}
	return std::ref(*procOstream);
}

std::ofstream& getThreadOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid == 0)
	{
		std::cout<<"Error, getThreadOstream() must be called in task's thread\n";
		std::ofstream output;
		return std::ref(output);
	}
	ThreadPrivateData *tpd = TaskBase::getThreadPrivateData();
	if(!tpd->threadOstream)
	{
		std::string filename = (TaskManager::getCurrentTask())->getName();
		filename.append("-thread");
		filename.append(std::to_string(TaskManager::getCurrentThreadRankinTask()));
		filename.append(".log");
		tpd->threadOstream = new std::ofstream(filename);
	}

	return std::ref(*(tpd->threadOstream));

}



}// namespcae iUtc
















