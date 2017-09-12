#include "TaskBase.h"
#include "TaskManager.h"
#include "UtcContext.h"
#include <map>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace iUtc{

boost::thread_specific_ptr<ThreadPrivateData> TaskBase::m_threadPrivateData;

std::string TaskBase::getName()
{
    return m_Name;
}

TaskType TaskBase::getTaskType(){

	return m_TaskType;
}

TaskId_t TaskBase::getTaskId()
{
    return m_TaskId;
}

TaskId_t TaskBase::getParentTaskId()
{
    return m_ParentTaskId;
}

TaskBase* TaskBase::getParentTask()
{
    TaskManager* mgr = UtcContext::getTaskManager();
    return mgr->getParentTask();
}

std::vector<ProcRank_t> TaskBase::getTaskRankList()
{
    return m_TaskRankList;
}

int TaskBase::getNumProcesses()
{
    return m_numProcesses;
}

int TaskBase::toLocal(ThreadRank_t trank)
{
	std::map<ThreadRank_t, int>::iterator it=m_ThreadRank2Local.find(trank);
	if(it == m_ThreadRank2Local.end())
	{
		std::cerr<<"Error for calling toLocal, no this thread rank\n";
		return -1;
	}
	return it->second;

}

int TaskBase:: getNumLocalThreads()
{
    return m_numLocalThreads;
}

std::vector<ThreadId_t> TaskBase::getLocalThreadList()
{
    return m_LocalThreadList;
}

int TaskBase::getNumTotalThreads()
{
    return m_numTotalThreads;
}

ProcRank_t TaskBase::getCurrentProcRank()
{
    return m_processRank;
}

ProcRank_t TaskBase::getProcRankOfThread(ThreadRank_t trank){
	return m_TaskRankList[trank];
}

bool TaskBase::isLocal(ThreadRank_t tRank)
{
    return m_TaskRankList[tRank]==m_processRank;
}

ThreadRank_t TaskBase::getThreadRankById(ThreadId_t tid)
{
	std::map<ThreadId_t, ThreadRank_t>::iterator it = m_LocalThreadRegistry.find(tid);
	if(it == m_LocalThreadRegistry.end())
	{
		std::cerr<<"Error for calling getThreadRankById, no this thread!"<<std::endl;
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
	m_threadPrivateData.reset(tpd);
}

bool TaskBase::isActiveOnCurrentProcess()
{
	// return true if task has threads mapped on current process
	return m_numLocalThreads!=0;
}

ProcRank_t TaskBase::getMainResideProcess()
{
	return m_mainResideProcess;
}

bool TaskBase::hasActiveLocalThread(){
	return false;
}

void TaskBase::waitLocalThreadFinish(){
	return;
}

#if ENABLE_SCOPED_DATA
	internal_MPIWin* TaskBase::getTaskMpiWindow(){
		return m_taskMpiInternalWindow;
	}
#endif

///
TaskBase::TaskBase()
:m_Name(""),
m_TaskId(-1),
m_TaskType(TaskType::unknown),
m_ParentTaskId(-1),
m_numProcesses(0),
m_numLocalThreads(0),
m_numTotalThreads(0),
m_processRank(-1),
m_procOstream(nullptr),
m_mainResideProcess(-1),
m_uniqueExeObjPtr(nullptr)
{
    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
    m_ThreadRank2Local.clear();
    m_bcastAvailable.store(0);
    m_gatherAvailable.store(0);

    m_worldRankToTaskGroupRank.clear();
    m_shmemSize = 0;
}

TaskBase::~TaskBase()
{

    m_TaskRankList.clear();
    m_LocalThreadList.clear();
    m_LocalThreadRegistry.clear();
    m_ThreadRank2Local.clear();
    m_procOstream=nullptr;
    if(m_uniqueExeObjPtr)
    	delete m_uniqueExeObjPtr;

    m_worldRankToTaskGroupRank.clear();
}

void TaskBase::RegisterTask()
{
    TaskManager* mgr = UtcContext::getTaskManager();  //TaskManager::getInstance()
    //TaskId id = mgr->registerTask(this);
    mgr->registerTask(this, m_TaskId);
    return;
}


void TaskBase::display()
{
	std::string tasktype[3]={"unknown", "cpu_task", "gpu_task"};
	std::string s;
	switch(m_TaskType){
	case TaskType::cpu_task:
		s = tasktype[1];
		break;
	case TaskType::gpu_task:
		s = tasktype[2];
		break;
	default:
		s = tasktype[0];
		break;
	}
	std::cout<<
			"Name:"<< m_Name<<
			", Type:"<<s<<
			", ProcessRank:"<<m_processRank<<
			", NumLocalThreads:"<<m_numLocalThreads<<
			", NumTotalThreads:"<<m_numTotalThreads<<
			std::endl;
	return;
}

} //namespace iUtc
