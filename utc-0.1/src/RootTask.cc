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
    m_TaskId = TaskManager::getNewTaskId();      // should get 0 for root
    ThreadId tid = TaskManager::getThreadId();   // main thread of current process,
                                                 // no other threads have been created yet
    m_LocalThreadList.push_back(tid);
    m_LocalThreadRegistry.insert(std::pair<ThreadId, ThreadRank>(tid, tRank));
    m_ThreadRank2Local.insert(std::pair<ThreadRank, int>(tRank, 0));
    m_ParentTaskId= m_TaskId; //only for root
    m_processRank= pRank;
    for(int i=0; i<WorldSize; i++)
    {
    	m_TaskRankList.push_back(i);
    }

#ifdef USE_DEBUG_LOG
    std::string filename= "Proc";
    filename.append(std::to_string(currentProcess));
    filename.append(".log");
    m_procOstream = new std::ofstream(filename);
#else
    m_procOstream = nullptr;
#endif

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
	if(m_procOstream)
	{
		if(m_procOstream->is_open())
		{
			m_procOstream->close();
		}
	}
    return;
}

std::ofstream* RootTask::getProcOstream()
{
	return m_procOstream;
}

void RootTask::setProcOstream(std::ofstream& procOstream)
{
	m_procOstream = &procOstream;
}

} //namespace iUtc
