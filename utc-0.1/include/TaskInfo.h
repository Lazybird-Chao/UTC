#ifndef UTC_TASKINFO_H_
#define UTC_TASKINFO_H_

#include <pthread.h>
#include <iostream>

#include "UtcBasics.h"
#include "Barrier.h"
#ifdef USE_MPI_BASE
	#include <mpi.h>
#endif

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
    TaskId  taskId = -1;   // same value in one task
    //the taskid of parent task
    TaskId  parentTaskId = -1;  // same value in one task
    //thread id of current thread
    ThreadId   threadId;    // diff value in each thread

    ThreadRank tRank = -1;  // thread rank in all threads that running for a task

    ProcRank pRank = -1;    // process rank in all processes that a task mapped to

    Barrier* barrierObjPtr = nullptr;   // same value in one task

#ifdef USE_MPI_BASE
    MPI_Comm* commPtr = nullptr;  // same value in one task
    MPI_Group* mpigroupPtr = nullptr; // same value in one task
#endif

 };

struct ThreadPrivateData
{
	 //
	 std::ofstream *threadOstream = nullptr;
};

}//namespace iUtc





#endif
