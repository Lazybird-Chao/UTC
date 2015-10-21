#include "RootTask.h"
#include "UtcBasics.h"
#include "TaskInfo.h"
#include "TaskManager.h"
#include "RankList.h"


#include <vector>
#include <cassert>
#include "boost/filesystem.hpp"

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
    assert(m_TaskId ==0);
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
    boost::filesystem::path log_path("./log");
    if(!exists(log_path))
    	boost::filesystem::create_directory(log_path);
    std::string filename= "./log/Proc";
    filename.append(std::to_string(currentProcess));
    filename.append(".log");
    m_procOstream = new std::ofstream(filename);
#else
    m_procOstream = nullptr;
#endif

    // create TaskInfo structure
    TaskInfo* taskInfoPtr = new TaskInfo();
    taskInfoPtr->pRank = pRank;
    taskInfoPtr->parentTaskId = m_ParentTaskId;
    taskInfoPtr->tRank = tRank;
    taskInfoPtr->taskId = m_TaskId;
    taskInfoPtr->threadId = tid;
    m_barrierObjPtr = new Barrier(1, 0);
    taskInfoPtr->barrierObjPtr = m_barrierObjPtr;
#ifdef USE_MPI_BASE
    MPI_Comm_group(MPI_COMM_WORLD, &m_mpigroup);
    m_comm = MPI_COMM_WORLD;
    taskInfoPtr->commPtr = &m_comm;
    taskInfoPtr->mpigroupPtr = &m_mpigroup;
#endif
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
#ifdef USE_DEBUG_LOG
			PRINT_TIME_NOW(*m_procOstream)
		    *m_procOstream<<"Root Task destroyed!!!"<<std::endl;
#endif
			m_procOstream->close();
		}
	}
	delete m_barrierObjPtr;
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

#ifdef USE_MPI_BASE
MPI_Comm* RootTask::getWorldComm()
{
	return &m_comm;
}

MPI_Group* RootTask::getWorldGroup()
{
	return &m_mpigroup;
}
#endif

} //namespace iUtc
