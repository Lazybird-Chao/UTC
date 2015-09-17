#include "TaskManager.h"
#include "UtcException.h"
#include <iostream>

namespace iUtc{

// initialize static variables
TaskManager* TaskManager::m_InstancePtr = nullptr;

std::map<TaskId, TaskBase*> TaskManager::m_TaskRegistry;

TaskId TaskManager::m_TaskIdDealer = 0;

std::mutex TaskManager::m_mutexTaskRegistry;
std::mutex TaskManager::m_mutexTaskIdDealer;

boost::thread_specific_ptr<TaskInfo> TaskManager::m_taskInfo;

TaskManager::TaskManager(){}

TaskManager::~TaskManager()
{
    if(m_InstancePtr)
    {
        m_TaskRegistry.clear();
        m_InstancePtr = nullptr;
        m_TaskIdDealer = 0;
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

void TaskManager::unregisterTask(TaskBase* task)
{
    std::lock_guard<std::mutex> lock(m_mutexTaskRegistry);
    TaskId id = task->getTaskId();
    if(id)
    {
        m_TaskRegistry.erase(id);
    }
    return;
}

TaskId TaskManager::getNewTaskId()
{
    std::lock_guard<std::mutex> lock(m_mutexTaskIdDealer);
    TaskId id = m_TaskIdDealer++;
    return id;
}


TaskInfo TaskManager::getTaskInfo(void)
{
    TaskInfo* taskInfo = m_taskInfo.get();

    if(taskInfo)
        return *taskInfo;
}

void TaskManager::setTaskInfo(TaskInfo* InfoPtr)
{
    m_taskInfo.reset(InfoPtr);
}

TaskId TaskManager::getCurrentTaskId()
{
    TaskId taskId = 0;
    TaskInfo* taskInfo = m_taskInfo.get();
    if(taskInfo)
    {
        taskId = taskInfo->taskId;
    }

    return taskId;
}

TaskId TaskManager::getParentTaskId()
{
    TaskId taskId = 0;
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




} // namespace iUtc


