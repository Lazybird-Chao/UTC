#include "Barrier.h"
#include "TaskManager.h"
#include "Task.h"
#include "UtcBasics.h"

#include <cassert>

namespace iUtc{

Barrier::Barrier()
{
	/*m_numLocalThreads = TaskManager::getCurrentTask()->getNumLocalThreads();
	m_intraThreadSyncCounterComing = 0;
	m_intraThreadSyncCounterLeaving = 0;
	m_taskId = TaskManager::getCurrentTaskId();*/


}
Barrier::Barrier(int numLocalThreads, int taskid)
{
	m_numLocalThreads = numLocalThreads;
	m_taskId = taskid;
	m_intraThreadSyncCounterComing = 0;
	m_intraThreadSyncCounterLeaving = 0;
}
#ifdef USE_MPI_BASE
Barrier::Barrier(int numLocalThreads, int taskid, MPI_Comm *comm)
{
	m_numLocalThreads = numLocalThreads;
	m_taskId = taskid;
	m_taskCommPtr = comm;
	m_intraThreadSyncCounterComing = 0;
	m_intraThreadSyncCounterLeaving = 0;
}
#endif

Barrier::~Barrier()
{
	m_intraThreadSyncCounterComing = 0;
	m_intraThreadSyncCounterLeaving = 0;
}

void Barrier::synch_intra()
{

	std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);
	// wait for all thread coming here
	m_intraThreadSyncCounterComing++;
	if(m_intraThreadSyncCounterComing == m_numLocalThreads)
	{
		// last coming thread do notify
		m_intraThreadSyncCond.notify_all();
		//
		m_intraThreadSyncCounterLeaving++;
		LCK1.unlock();
	}
	else
	{
		// early coming thread do wait
		m_intraThreadSyncCond.wait(LCK1,
					[=](){return m_intraThreadSyncCounterComing == m_numLocalThreads;});
		//
		m_intraThreadSyncCounterLeaving++;
		if(m_intraThreadSyncCounterLeaving==m_numLocalThreads)
		{
			// last leaving thread reset counter value
			m_intraThreadSyncCounterComing = 0;
			m_intraThreadSyncCounterLeaving = 0;
		}
		LCK1.unlock();
	}

}

void Barrier::synch_inter()
{
	std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);
	// wait for all thread coming here
	m_intraThreadSyncCounterComing++;
	if(m_intraThreadSyncCounterComing == m_numLocalThreads)
	{
		// last coming thread
		// before notify, wait for other processes coming to this point
#ifdef USE_MPI_BASE
		// do barrier across the processes that this task mapped to
		MPI_Barrier(*m_taskCommPtr);
#endif
		//do notify
		m_intraThreadSyncCond.notify_all();
		//
		assert(m_intraThreadSyncCounterLeaving == 0);
		m_intraThreadSyncCounterLeaving++;
		LCK1.unlock();
	}
	else
	{
		// early coming thread do wait
		m_intraThreadSyncCond.wait(LCK1,
					[=](){return m_intraThreadSyncCounterComing == m_numLocalThreads;});
		//
		m_intraThreadSyncCounterLeaving++;
		if(m_intraThreadSyncCounterLeaving==m_numLocalThreads)
		{
			// last leaving thread reset counter value
			m_intraThreadSyncCounterComing = 0;
			m_intraThreadSyncCounterLeaving = 0;
		}
		LCK1.unlock();
	}

}

void intra_Barrier()
{
	static thread_local Barrier *taskBarrierPtr  = nullptr;
	if(!taskBarrierPtr)
		taskBarrierPtr = TaskManager::getTaskInfo()->barrierObjPtr;

#ifdef USE_DEBUG_LOG
	std::ofstream *m_threadOstream = getThreadOstream();
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" comes to intra sync point."<<std::endl;
#endif

	taskBarrierPtr->synch_intra();

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" leaves intra sync point."<<std::endl;
#endif
}

void inter_Barrier()
{
	static thread_local Barrier *taskBarrierPtr  = nullptr;
	if(!taskBarrierPtr)
		taskBarrierPtr = TaskManager::getTaskInfo()->barrierObjPtr;

#ifdef USE_DEBUG_LOG
	std::ofstream *m_threadOstream = getThreadOstream();
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" comes to inter sync point."<<std::endl;
#endif

	taskBarrierPtr->synch_inter();

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" leaves inter sync point."<<std::endl;
#endif

}


}// namepsce iUtc
