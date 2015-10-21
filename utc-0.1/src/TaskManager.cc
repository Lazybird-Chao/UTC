#include "TaskManager.h"
#include "UtcException.h"
#include <iostream>
#include <cassert>

namespace iUtc{

// initialize static variables
TaskManager* TaskManager::m_InstancePtr = nullptr;

std::map<TaskId, TaskBase*> TaskManager::m_TaskRegistry;

TaskId TaskManager::m_TaskIdDealer = 0;

RootTask* TaskManager::m_root = nullptr;

std::mutex TaskManager::m_mutexTaskRegistry;
std::mutex TaskManager::m_mutexTaskIdDealer;

boost::thread_specific_ptr<TaskInfo> TaskManager::m_taskInfo;

std::ofstream* getProcOstream();

TaskManager::TaskManager(){}

TaskManager::~TaskManager()
{
    if(m_InstancePtr)
    {
        assert(m_TaskRegistry.size() ==0);
        m_TaskRegistry.clear();
        m_InstancePtr = nullptr;
        m_TaskIdDealer = 0;
#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"TaskManager destroyed!!!"<<std::endl;
#endif
        m_root = nullptr;
        m_taskInfo.reset();      // Be carefull of this!
    }


}

TaskManager* TaskManager::getInstance()
{
    // Singleton instance
    if( !m_InstancePtr)
    {
        // TODO: using c++11 local static variable feature to ensure thread safe
        /*static TaskManager taskMgr;
        m_InstancePtr = &taskMgr;*/
        m_InstancePtr = new TaskManager(); // if multi thread call this, may cause problem
    }
    return m_InstancePtr;
}

TaskId TaskManager::registerTask(TaskBase* task)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);
    TaskId id = task->getTaskId();
    m_TaskRegistry.insert(std::pair<TaskId, TaskBase*>(id, task));
    return id;
}
void TaskManager::registerTask(TaskBase* task, int id)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);
    m_TaskRegistry.insert(std::pair<TaskId, TaskBase*>(id, task));
    return;
}

void TaskManager::unregisterTask(TaskBase* task)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);
    TaskId id = task->getTaskId();
    if(id)
    {	// root will not be erased, root task id is 0
        m_TaskRegistry.erase(id);     // should check if in the map
    }
    return;
}
void TaskManager::unregisterTask(TaskBase* task, int id)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);

    // root will not be erased, root task id is 0
    m_TaskRegistry.erase(id);        // should check if in the map
    return;
}

bool TaskManager::hasTaskItem(int id)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);
    if(m_TaskRegistry.find(id) == m_TaskRegistry.end())
    {
        return false;
    }
    else
        return true;
}


TaskId TaskManager::getNewTaskId()
{
    std::lock_guard<std::mutex> lock(m_mutexTaskIdDealer);
    TaskId id = m_TaskIdDealer++;
    return id;
}

int TaskManager::getNumTasks()
{
	// will include the root task on each process
	return m_TaskRegistry.size();
}


TaskInfo* TaskManager::getTaskInfo(void)
{
    TaskInfo* taskInfo = m_taskInfo.get();

    return taskInfo;
}

void TaskManager::setTaskInfo(TaskInfo* InfoPtr)
{
	if(!InfoPtr)
		m_taskInfo.reset();
	else
		m_taskInfo.reset(InfoPtr);
}

TaskId TaskManager::getCurrentTaskId()
{
    TaskId taskId = -1;
    TaskInfo* taskInfo = m_taskInfo.get();
    if(taskInfo)
    {
        taskId = taskInfo->taskId;
    }

    return taskId;
}

TaskId TaskManager::getParentTaskId()
{
    TaskId taskId = -1;
    TaskInfo* taskInfo = m_taskInfo.get();
    if(taskInfo)
    {
        taskId = taskInfo->parentTaskId;
    }

    return taskId;
}

TaskBase* TaskManager::getCurrentTask()
{
    return m_TaskRegistry[getCurrentTaskId()];

}

TaskBase* TaskManager::getParentTask()
{
    return m_TaskRegistry[getParentTaskId()];
}

ThreadId TaskManager::getThreadId()
{
    return std::this_thread::get_id();
}

ThreadRank TaskManager::getCurrentThreadRankinTask()
{
    TaskInfo* taskInfo = m_taskInfo.get();
    return taskInfo->tRank;
}

ProcRank TaskManager::getCurrentProcessRankinTask()
{
    TaskInfo* taskInfo = m_taskInfo.get();
    return taskInfo->pRank;
}

#ifdef USE_MPI_BASE
MPI_Comm* TaskManager::getCurrentTaskComm()
{
	TaskInfo* taskInfo = m_taskInfo.get();
	return taskInfo->commPtr;
}

MPI_Group* TaskManager::getCurrentTaskmpiGroup()
{
	TaskInfo* taskInfo = m_taskInfo.get();
	return taskInfo->mpigroupPtr;
}
#endif


void TaskManager::setRootTask(RootTask* root)
{
	m_root = root;
}

RootTask* TaskManager::getRootTask()
{
	return m_root;
}


} // namespace iUtc


