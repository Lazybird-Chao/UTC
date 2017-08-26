#include "RootTask.h"
#include "UtcBasics.h"
#include "TaskInfo.h"
#include "TaskManager.h"
#include <vector>
#include <cassert>
#include "boost/filesystem.hpp"
#include "../include/ProcList.h"

namespace iUtc{

RootTask::RootTask(int WorldSize, int currentProcess)
:TaskBase()
{
    m_numProcesses = WorldSize;
    m_numLocalThreads = 1;
    m_numTotalThreads = m_numProcesses;
    m_Name = "RootTask";
    ThreadRank_t tRank = currentProcess;
    ProcRank_t pRank = currentProcess;
    m_TaskId = TaskManager::getNewTaskId();      // should get 0 for root
#ifdef USE_DEBUG_ASSERT
    assert(m_TaskId ==0);
#endif
    ThreadId_t tid = TaskManager::getThreadId();   // main thread of current process,
                                                 // no other threads have been created yet
    m_LocalThreadList.push_back(tid);
    m_LocalThreadRegistry.insert(std::pair<ThreadId_t, ThreadRank_t>(tid, tRank));
    m_ThreadRank2Local.insert(std::pair<ThreadRank_t, int>(tRank, 0));
    m_ParentTaskId= m_TaskId; //only for root
    m_processRank= pRank;
    m_mainResideProcess = 0;
    for(int i=0; i<WorldSize; i++)
    {
    	m_TaskRankList.push_back(i);
    }

#ifdef USE_DEBUG_LOG
    boost::filesystem::path log_path("./log");
    if(!boost::filesystem::exists(log_path))
    	boost::filesystem::create_directory(log_path);
    std::string filename= "./log/Proc";
    filename.append(std::to_string(currentProcess));
    filename.append(".log");
    m_procOstream = new std::ofstream(filename);
#else
    m_procOstream = nullptr;
#endif


#ifdef USE_MPI_BASE
    /*
    MPI_Comm_group(MPI_COMM_WORLD, &m_worldGroup);
    m_worldComm = MPI_COMM_WORLD;
    */
    MPI_Comm_dup(MPI_COMM_WORLD, &m_worldComm);
    MPI_Comm_group(m_worldComm, &m_worldGroup);
    for(int i=0; i<WorldSize; i++)
       m_worldRankToTaskGroupRank.insert(std::pair<int, int>(i,i));
#endif
    // create TaskInfo structure
    TaskInfo* taskInfoPtr = new TaskInfo();
    taskInfoPtr->pRank = pRank;
    taskInfoPtr->parentTaskId = m_ParentTaskId;
    taskInfoPtr->tRank = tRank;
    taskInfoPtr->lRank = 0;
    taskInfoPtr->taskId = m_TaskId;
    taskInfoPtr->threadId = tid;
    m_barrierObjPtr = new Barrier(1, 0, &m_worldComm);
    taskInfoPtr->barrierObjPtr = m_barrierObjPtr;
#ifdef USE_MPI_BASE
    taskInfoPtr->commPtr = &m_worldComm;
    taskInfoPtr->mpigroupPtr = &m_worldGroup;

    taskInfoPtr->worldCommPtr = &m_worldComm;
    taskInfoPtr->worldGroupPtr = &m_worldGroup;
    taskInfoPtr->worldRankToGrouprRank = &m_worldRankToTaskGroupRank;
#endif

    TaskManager::setTaskInfo(taskInfoPtr);    // reside in main thread of current process, same as
                                              // TaskManager instance, only one instance in current
                                              // process.
    MPI_Barrier(m_worldComm);
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
			//m_procOstream->close();
		}
	}
	delete m_barrierObjPtr;

	//
	MPI_Barrier(m_worldComm);
	MPI_Comm_free(&m_worldComm);
	MPI_Group_free(&m_worldGroup);

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
	return &m_worldComm;
}

MPI_Group* RootTask::getWorldGroup()
{
	return &m_worldGroup;
}
#endif

} //namespace iUtc
