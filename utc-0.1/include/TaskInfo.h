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

    ThreadRank tRank;  // thread rank in all threads that running for a task

    ProcRank pRank;    // process rank in all processes that a task mapped to

 };

struct ThreadPrivateData
{
	 //
	 std::ofstream *threadOstream;
};

}//namespace iUtc





#endif
