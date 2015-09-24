#ifndef TASKMANAGER_H_
#define TASKMANAGER_H_

#include "UtcBasics.h"
#include "TaskBase.h"
#include "TaskInfo.h"
#include <map>
#include <mutex>
#include <thread>
#include <boost/thread/tss.hpp>
#include <boost/thread/thread.hpp>

namespace iUtc{
    /**
     *  TaskManager is a singleton that holds all the tasks in a process
     */
    class TaskManager
    {
        public:
            //the sole instance of the class
            static TaskManager* getInstance();

            static TaskId registerTask(TaskBase* task);

            static void unregisterTask(TaskBase* task);

            static TaskId getNewTaskId();

            // these calls are TSS data related, when calling in main thread(or
            // process), it's actually root task's TSS info
            static TaskInfo getTaskInfo(void);
            static void setTaskInfo(TaskInfo* InfoPtr);

            static TaskId getCurrentTaskId();

            static TaskId getParentTaskId();

            static TaskBase* getCurrentTask();

            static TaskBase* getParentTask();

            //
            static ThreadId getThreadId();

            static ThreadRank getCurrentThreadRankinTask();

            static ProcRank getCurrentProcessRankinTask();

            static void setRootTask(TaskBase *root);

            static TaskBase* getRootTask();

            ~TaskManager();

        protected:

            //task registry
            static std::map<TaskId, TaskBase*> m_TaskRegistry;

            // forbidden to use
            TaskManager();
            TaskManager(const TaskManager&);
            TaskManager& operator=(const TaskManager&);

        private:
            //the pointer to the sole instances
            static TaskManager* m_InstancePtr;

            //task id dealer
            static TaskId m_TaskIdDealer;


            static std::mutex m_mutexTaskRegistry;
            static std::mutex m_mutexTaskIdDealer;

            static boost::thread_specific_ptr<TaskInfo> m_taskInfo;

            static TaskBase *m_root;

    };//class TaskManager
}// namespace iUtc




#endif

