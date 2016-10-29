#ifndef UTC_TASKINFO_H_
#define UTC_TASKINFO_H_

#include "UtcBasics.h"
#include "Barrier.h"
#include "UniqueExeTag.h"


#include <pthread.h>
#include <atomic>
#include <iostream>

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

/* cpu task specific info */
 struct CPUTaskSpecInfo{

 };


 /* gpu task specific info*/
 struct GPUTaskSpecInfo{
	 // current thread's binded GPU device Id
	 int gpuId=-1;

 };


 struct TaskInfo
 {
    //the task id that the current thread belongs to
    TaskId_t  taskId = -1;   // same value in one task
    //the taskid of parent task
    TaskId_t  parentTaskId = -1;  // same value in one task
    //thread id of current thread
    ThreadId_t   threadId;    // diff value in each thread

    ThreadRank_t tRank = -1;  // thread rank in all threads that running for a task

    ProcRank_t pRank = -1;    // process rank in all processes that a task mapped to

    ThreadRank_t lRank = -1; // local thread rank of a task in one process

    Barrier* barrierObjPtr = nullptr;   // same value in one task
    SpinBarrier* spinBarrierObjPtr = nullptr;

#ifdef USE_MPI_BASE
    MPI_Comm* commPtr = nullptr;  // same value in one task
    MPI_Group* mpigroupPtr = nullptr; // same value in one task
#endif

    struct CPUTaskSpecInfo cpuSpecInfo;

    struct GPUTaskSpecInfo gpuSpecInfo;

 };



struct ThreadPrivateData
{
	 //
	 std::ofstream *threadOstream = nullptr;
	 UniqueExeTag *taskUniqueExeTagObj = nullptr;
	 //
	 std::atomic<int> *bcastAvailable;
	 std::atomic<int> *gatherAvailable;
};

}//namespace iUtc





#endif
