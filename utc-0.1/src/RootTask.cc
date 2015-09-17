#include "RootTask.h"
#include "UtcBasics.h"
#include "TaskInfo.h"
#include "TaskManager.h"
#include "RankList.h"


#include <vector>

namespace iUtc{

RootTask::RootTask(int WorldSize, int currentProcess)
:TaskBase()
{
    m_numProcesses = WorldSize;
    m_numLocalThreads = 1;
    m_Name = "RootTask";
    ThreadRank tRank = 0;
    ProcRank pRank = currentProcess;
    m_TaskId = TaskManager::getNewTaskId();
    ThreadId tid = TaskManager::getThreadId();   // main thread of current process,
                                                 // no other threads have been created yet
    m_LocalThreadList.push_back(tid);
    m_LocalThreadRegistry.insert(std::pair<ThreadId, ThreadRank>(tid, tRank));
    m_ParentTaskId= m_TaskId; //only for root
    m_processRank= pRank;
    RankList rlist(WorldSize);
    rlist.getRankListVector(m_TaskMapList);


    TaskInfo* taskInfoPtr = new TaskInfo();
    taskInfoPtr->pRank = pRank;
    taskInfoPtr->parentTaskId = m_ParentTaskId;
    taskInfoPtr->tRank = tRank;
    taskInfoPtr->taskId = m_TaskId;
    taskInfoPtr->threadId = tid;
    TaskManager::setTaskInfo(taskInfoPtr);    // reside in main thread of current process, same as
                                              // TaskManager instance, only one instance in current
                                              // process.



}


RootTask::~RootTask()
{
    return;
}



} //namespace iUtc
