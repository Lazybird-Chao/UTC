#include "TaskBase.h"
#include "TaskManager.h"
#include "UtcContext.h"
#include <map>

namespace iUtc{

boost::thread_specific_ptr<ThreadPrivateData> TaskBase::m_threadPrivateData;

std::string TaskBase::getName()
{
    return m_Name;
}

TaskId TaskBase::getTaskId()
{
    return m_TaskId;
}

TaskId TaskBase::getParentTaskId()
{
    return m_ParentTaskId;
}

TaskBase* TaskBase::getParentTask()
{
    TaskManager* mgr = UtcContext::getTaskManager();
    return mgr->getParentTask();
}

std::vector<ProcRank> TaskBase::getTaskRankList()
{
    return m_TaskRankList;
}

int TaskBase::getNumProcesses()
{
    return m_numProcesses;
}

int TaskBase::toLocal(ThreadRank trank)
{
	std::map<ThreadRank, int>::iterator it=m_ThreadRank2Local.find(trank);
	if(it == m_ThreadRank2Local.end())
	{
		std::cout<<"Error for calling toLocal, no this thread rank\n";
		return -1;
	}
	return it->second;

}

int TaskBase:: getNumLocalThreads()
{
    return m_numLocalThreads;
}

std::vector<ThreadId> TaskBase::getLocalThreadList()
{
    return m_LocalThreadList;
}

int TaskBase::getNumTotalThreads()
{
    return m_numTotalThreads;
}

ProcRank TaskBase::getCurrentProcRank()
{
    return m_processRank;
}

bool TaskBase::isLocal(ThreadRank tRank)
{
    return m_TaskRankList[tRank]==m_processRank;
}

ThreadRank TaskBase::getThreadRankById(ThreadId tid)
{
	std::map<ThreadId, ThreadRank>::iterator it = m_LocalThreadRegistry.find(tid);
	if(it == m_LocalThreadRegistry.end())
	{
		std::cout<<"Error for calling getThreadRankById, no this thread\n";
		return -1;
	}
    return it->second; //m_LocalThreadRegistry[tid];
}

ThreadPrivateData* TaskBase::getThreadPrivateData()
{
	ThreadPrivateData *tpd = m_threadPrivateData.get();
	return tpd;
}

void TaskBase::setThreadPrivateData(ThreadPrivateData * tpd)
{
	m_threadPrivateData.reset();
}

///
TaskBase::TaskBase()
:m_Name(""),
m_TaskId(-1),
m_ParentTaskId(-1),
m_numProcesses(0),
m_numLocalThreads(0),
m_numTotalThreads(0),
m_processRank(-1),
m_procOstream(nullptr)
{
    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
    m_ThreadRank2Local.clear();
}

TaskBase::~TaskBase()
{

    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
    m_ThreadRank2Local.clear();
}

TaskId TaskBase::RegisterTask()
{
    TaskManager* mgr = UtcContext::getTaskManager();  //TaskManager::getInstance()
    TaskId id = mgr->registerTask(this);
    return id;
}

} //namespace iUtc
