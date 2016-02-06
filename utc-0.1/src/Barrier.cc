#include "Barrier.h"
#include "TaskManager.h"
#include "RootTask.h"
#include "Task.h"
#include "UtcBasics.h"


#include <cassert>
#include "../include/TaskUtilities.h"

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
#ifdef USE_MPI_BASE
	RootTask *root = TaskManager::getRootTask();
	m_taskCommPtr = root->getWorldComm();
#endif
	m_intraThreadSyncCounterComing[0] = m_intraThreadSyncCounterComing[1]=0;
	m_intraThreadSyncCounterLeaving[0] = m_intraThreadSyncCounterLeaving[1] = 0;
	m_countIdx = (int*)malloc(sizeof(int)*numLocalThreads);
	for(int i=0; i<numLocalThreads; i++)
		m_countIdx[i] = 0;
}


Barrier::Barrier(int numLocalThreads, int taskid, MPI_Comm* comm)
{
	m_numLocalThreads = numLocalThreads;
	m_taskId = taskid;
#ifdef USE_MPI_BASE
	m_taskCommPtr = comm;
#endif
	m_intraThreadSyncCounterComing[0] = m_intraThreadSyncCounterComing[1]=0;
	m_intraThreadSyncCounterLeaving[0] = m_intraThreadSyncCounterLeaving[1] = 0;
	m_countIdx = (int*)malloc(sizeof(int)*numLocalThreads);
	for(int i=0; i<numLocalThreads; i++)
		m_countIdx[i] = 0;

}


Barrier::~Barrier()
{
	m_intraThreadSyncCounterComing[0] = m_intraThreadSyncCounterComing[1]=0;
	m_intraThreadSyncCounterLeaving[0] = m_intraThreadSyncCounterLeaving[1] = 0;
	if(m_countIdx)
		free(m_countIdx);
}

void Barrier::synch_intra(int local_rank)
{
	std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);

	// wait for all thread coming here
	m_intraThreadSyncCounterComing[m_countIdx[local_rank]]++;
	if(m_intraThreadSyncCounterComing[m_countIdx[local_rank]] == m_numLocalThreads)
	{
		// last coming thread do notify
		m_intraThreadSyncCond.notify_all();
#ifdef USE_DEBUG_ASSERT
		assert(m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]] == 0);
#endif
	}
	else
	{
		// early coming thread do wait
		m_intraThreadSyncCond.wait(LCK1,
					[=](){return m_intraThreadSyncCounterComing[m_countIdx[local_rank]] == m_numLocalThreads;});
		//
	}

	m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]]++;
	if(m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]]==m_numLocalThreads)
	{
		// last leaving thread reset counter value
		m_intraThreadSyncCounterComing[m_countIdx[local_rank]] = 0;
		m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]] = 0;

	}
	// rotate idx value to use another counter for next barrier op
	m_countIdx[local_rank] = (m_countIdx[local_rank]+1)%2;
	LCK1.unlock();

}

void Barrier::synch_inter(int local_rank)
{
	std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);
	// wait for all local thread coming here
	m_intraThreadSyncCounterComing[m_countIdx[local_rank]]++;
	if(m_intraThreadSyncCounterComing[m_countIdx[local_rank]] == m_numLocalThreads)
	{
		// last coming thread
		// before notify, wait for other processes coming to this point
#ifdef USE_MPI_BASE
		// do barrier across the processes that this task mapped to
		MPI_Barrier(*m_taskCommPtr);

#endif
		//do notify
		m_intraThreadSyncCond.notify_all();
#ifdef USE_DEBUG_ASSERT
		assert(m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]] == 0);
#endif

	}
	else
	{
		// early coming thread do wait
		m_intraThreadSyncCond.wait(LCK1,
					[=](){return m_intraThreadSyncCounterComing[m_countIdx[local_rank]] == m_numLocalThreads;});
	}

	//
	m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]]++;
	if(m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]]==m_numLocalThreads)
	{
		// last leaving thread reset counter value
		m_intraThreadSyncCounterComing[m_countIdx[local_rank]] = 0;
		m_intraThreadSyncCounterLeaving[m_countIdx[local_rank]] = 0;
	}
	m_countIdx[local_rank] = (m_countIdx[local_rank]+1)%2;
	LCK1.unlock();


}


void intra_Barrier()
{
	static thread_local Barrier *taskBarrierPtr  = nullptr;
	if(!taskBarrierPtr)
		taskBarrierPtr = TaskManager::getTaskInfo()->barrierObjPtr;
	static thread_local int local_rank =-1;
	if(local_rank ==-1)
		//local_rank = TaskManager::getCurrentTask()->toLocal(TaskManager::getCurrentThreadRankinTask());
		local_rank = TaskManager::getCurrentThreadRankInLocal();
#ifdef USE_DEBUG_LOG
	std::ofstream *m_threadOstream = getThreadOstream();
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" comes to intra sync point."<<local_rank<<std::endl;
#endif

	taskBarrierPtr->synch_intra(local_rank);

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
	static thread_local int local_rank =-1;
		if(local_rank ==-1)
			local_rank = TaskManager::getCurrentTask()->toLocal(TaskManager::getCurrentThreadRankinTask());

#ifdef USE_DEBUG_LOG
	std::ofstream *m_Ostream;
	if(TaskManager::getCurrentTaskId() == 0)
		 m_Ostream= getProcOstream();
	else
		m_Ostream = getThreadOstream();
	PRINT_TIME_NOW(*m_Ostream)
	*m_Ostream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" comes to inter sync point."<<local_rank<<std::endl;
#endif

	taskBarrierPtr->synch_inter(local_rank);

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_Ostream)
	*m_Ostream<<"thread "<<TaskManager::getCurrentThreadRankinTask()<<
				" leaves inter sync point."<<std::endl;
#endif

}


}// namepsce iUtc
