#ifndef TASKMANAGER_H_
#define TASKMANAGER_H_

#include "UtcBasics.h"
#include "TaskBase.h"
#include "TaskInfo.h"
#include "RootTask.h"
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

            static TaskId_t registerTask(TaskBase* task);
            static void registerTask(TaskBase* task, int id);

            static void unregisterTask(TaskBase* task);
            static void unregisterTask(TaskBase* task, int id);

            static bool hasTaskItem(int taskid);

            static TaskId_t getNewTaskId();

            static int getNumTasks();

            // these methods are TSS data related, when calling in main thread(or
            // process), it's actually root task's TSS info
            static TaskInfo* getTaskInfo(void);
            static void setTaskInfo(TaskInfo* InfoPtr);

            static TaskId_t getCurrentTaskId();

            static TaskId_t getParentTaskId();

            static TaskBase* getCurrentTask();

            static TaskBase* getParentTask();

            static ThreadId_t getThreadId();

            static ThreadRank_t getCurrentThreadRankinTask();

            static ProcRank_t getCurrentProcessRankinTask();

            static ThreadRank_t getCurrentThreadRankInLocal();

#ifdef USE_MPI_BASE
            static MPI_Comm* getCurrentTaskComm();
            static MPI_Group* getCurrentTaskmpiGroup();
#endif

            //
            static void setRootTask(RootTask *root);
            static RootTask* getRootTask();


            ~TaskManager();

        protected:

            //task registry
            static std::map<TaskId_t, TaskBase*> m_TaskRegistry;

            // forbidden to use
            TaskManager();
            TaskManager(const TaskManager&);
            TaskManager& operator=(const TaskManager&);

        private:
            //the pointer to the sole instances
            static TaskManager* m_InstancePtr;

            //task id dealer
            static TaskId_t m_TaskIdDealer;


            static std::mutex m_mutexTaskRegistry;
            static std::mutex m_mutexTaskIdDealer;

            static boost::thread_specific_ptr<TaskInfo> m_taskInfo;

            static RootTask *m_root;

    };//class TaskManager
}// namespace iUtc




#endif

