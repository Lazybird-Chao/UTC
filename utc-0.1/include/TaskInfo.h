#ifndef UTC_TASKINFO_H_
#define UTC_TASKINFO_H_

#include <pthread.h>
#include <iostream>
#include "UtcBasics.h"

namespace iUtc{

/**
 * \brief TaskInfo is a struct that is used for Thread Local storage within the Task Class
 *
 * \remarks This class is not designed to be used outside of the Task class.
 *
 */
 struct TaskInfo
 {
    //the task id that the current thread belongs to
    TaskId  taskId;
    //the taskid of parent task
    TaskId  parentTaskId;
    //thread id of current thread
    ThreadId   threadId;

    ThreadRank tRank;  // localRank

    ProcRank pRank;    // globalRank

 };

//std::ostream& operator<< (std::ostream& output, const TaskInfo& taskInfo);

}//namespace iUtc





#endif
