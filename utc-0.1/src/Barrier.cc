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
	m_numLocalThreads = 0;
	m_intraThreadSyncCounterComing[0] = m_intraThreadSyncCounterComing[1]=0;
	m_intraThreadSyncCounterLeaving[0] = m_intraThreadSyncCounterLeaving[1] = 0;
	m_taskId = -1;
	m_countIdx = nullptr;
	m_taskCommPtr =nullptr;
	m_threadSyncBarrier = nullptr;
	m_generation = m_counter =0;
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
	m_threadSyncBarrier = new boost::barrier(numLocalThreads);
	m_generation = m_counter =0;
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
	m_threadSyncBarrier = new boost::barrier(numLocalThreads);
	m_generation = m_counter =0;
}


Barrier::~Barrier()
{
	m_intraThreadSyncCounterComing[0] = m_intraThreadSyncCounterComing[1]=0;
	m_intraThreadSyncCounterLeaving[0] = m_intraThreadSyncCounterLeaving[1] = 0;
	if(m_countIdx)
		free(m_countIdx);
	if(m_threadSyncBarrier)
		delete m_threadSyncBarrier;
}

void Barrier::synch_intra(int local_rank)
{
	/*std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);

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
	LCK1.unlock();*/

	m_threadSyncBarrier->count_down_and_wait();

}

void Barrier::synch_inter(int local_rank)
{
	std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);
	int generation = m_generation;
	m_counter++;
	if(m_counter==m_numLocalThreads){
		// last coming thread
		// before notify, wait for other processes coming to this point
#ifdef USE_MPI_BASE
		// do barrier across the processes that this task mapped to
		MPI_Barrier(*m_taskCommPtr);

#endif
		m_counter=0;
		m_generation++;
		//do notify
		m_intraThreadSyncCond.notify_all();
		return;
	}
	while(generation == m_generation){
		m_intraThreadSyncCond.wait(LCK1);
	}
	return;
	/*std::unique_lock<std::mutex> LCK1(m_intraThreadSyncMutex);
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
	LCK1.unlock();*/


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


SpinBarrier::SpinBarrier(){
	m_numThreadsForSync =0;
	m_barrierCounter=0;
	m_generation=0;
	m_barrierReady=0;
}

SpinBarrier::SpinBarrier(int nthreads){
	m_numThreadsForSync = nthreads;
	m_barrierCounter=0;
	m_generation=0;
	m_barrierReady=nthreads;
}

void SpinBarrier::set(int nthreads){
	if(m_numThreadsForSync!=nthreads){
		m_barrierReady.store(nthreads);
		m_numThreadsForSync = m_barrierReady;
	}
}

void SpinBarrier::wait(){
	// wait set() finish
	long _counter=0;
	while(m_barrierReady.load() != m_numThreadsForSync){
		_counter++;
		if(_counter<USE_PAUSE)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP){
			__asm__ __volatile__ ("pause" ::: "memory");
			std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);
	}

	int generation = m_generation.load();
	int threadsForSync = m_numThreadsForSync;
	//
	m_barrierCounter.fetch_add(1);
	if(m_barrierCounter.compare_exchange_strong(threadsForSync, 0, std::memory_order_release)){
		m_generation.fetch_add(1);
		return;
	}
	long _counter=0;
	while(m_generation.load()==generation){
		_counter++;
		if(_counter<USE_PAUSE)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP){
			__asm__ __volatile__ ("pause" ::: "memory");
			std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);

	}
	return;
}


void intra_SpinBarrier(){
	static thread_local SpinBarrier* taskSpinBarrierPtr = nullptr;
	if(taskSpinBarrierPtr==nullptr){
		taskSpinBarrierPtr = TaskManager::getTaskInfo()->spinBarrierObjPtr;
	}
	taskSpinBarrierPtr->wait();
}


}// namepsce iUtc
