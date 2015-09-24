#include "TaskBase.h"
#include "TaskManager.h"
#include "UtcContext.h"


namespace iUtc{

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
    return m_LocalThreadRegistry[tid];
}

///
TaskBase::TaskBase()
:m_Name(""),
m_TaskId(-1),
m_ParentTaskId(-1),
m_numProcesses(0),
m_numLocalThreads(0),
m_numTotalThreads(0),
m_processRank(-1)
{

    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
}

TaskBase::~TaskBase()
{
    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
}

TaskId TaskBase::RegisterTask()
{
    TaskManager* mgr = UtcContext::getTaskManager();  //TaskManager::getInstance()
    TaskId id = mgr->registerTask(this);
    return id;
}

} //namespace iUtc
