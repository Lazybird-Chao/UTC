#include "XprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "Task_Utilities.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>

namespace iUtc{

thread_local std::ofstream *XprocConduit::m_threadOstream = nullptr;

XprocConduit::XprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId)
:ConduitBase()
{
	m_conduitId = cdtId;
	m_srcTask = srctask;
	m_dstTask = dsttask;
	m_srcId = m_srcTask->getTaskId();
	m_dstId = m_dstTask->getTaskId();

	m_numSrcLocalThreads = srctask->getNumLocalThreads();
	m_numDstLocalThreads = dsttask->getNumLocalThreads();

	m_srcMainResideProc = srctask->getMainResideProcess();
	m_dstMainResideProc = dsttask->getMainResideProcess();

	m_capacity = 16; // This may be a bug point. TODO:

	m_availableNoFinishedReadOpCount = m_capacity;
	m_availableNoFinishedWriteOpCount = m_capacity;
	m_WriteOpRotateCounter = new int[m_capacity+1];
	m_ReadOpRotateCounter = new int[m_capacity +1];
	m_WriteOpRotateFinishFlag = new int[m_capacity+1];
	m_ReadOpRotateFinishFlag = new int[m_capacity+1];
	for(int i =0;i<m_capacity+1; i++)
	{
		m_WriteOpRotateCounter[i]=0;
		m_ReadOpRotateCounter[i]=0;
		m_WriteOpRotateFinishFlag[i]=0;
		m_ReadOpRotateFinishFlag[i]=0;
	}
	//actually only use the local threads in one process, other thread pos is not used
	m_WriteOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	m_ReadOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	for(int i =0; i<m_srcTask->getNumTotalThreads();i++)
	{
		m_WriteOpRotateCounterIdx[i] = 0;
		m_ReadOpRotateCounterIdx[i]=0;
	}

	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();


#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"XprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] initiated..."<<std::endl;
#endif
}

int XprocConduit::Write(void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    //
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}


	std::unique_lock<std::mutex> LCK1(m_WriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif
	int counteridx = m_WriteOpRotateCounterIdx[myThreadRank];
	m_WriteOpRotateCounter[counteridx]++;
	if(m_WriteOpRotateCounter[counteridx]>1)
	{
		// a late thread
#ifdef USE_DEBUG_ASSERT
		assert(m_WriteOpRotateCounter[counteridx] <= localNumthreads);
#endif
		while(m_WriteOpRotateFinishFlag[counteridx] == 0)
		{
			m_WriteOpFinishCond.wait(LCK1);
		}
		// wake up
		m_WriteOpRotateFinishFlag[counteridx]++;
		if(m_WriteOpRotateFinishFlag[counteridx]==localNumthreads)
		{
			//last leaving thread
			m_WriteOpRotateFinishFlag[counteridx] = 0;
			m_WriteOpRotateCounter[counteridx] = 0;
			//
			m_availableNoFinishedWriteOpCount++;
			m_availableNoFinishedWriteCond.notify_one();

			// update counter idx to next one
			m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
			LCK1.unlock();
		}
		else
		{
			// update counter idx to next one
			m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
			LCK1.unlock();
		}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit write:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit write:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

		return 0;
	}
	else
	{
		// first thread
		while(m_availableNoFinishedWriteOpCount == 0)
		{
			m_availableNoFinishedWriteCond.wait(LCK1);
		}
		m_availableNoFinishedWriteOpCount--;
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		// because in one proc, there may be several tasks, so two different mpi msg
	    // could have same src/dst proc, if these two msg send by different tasks use
	    // same tag, then mpi msg would has same msg-envelop, may cause msg matching
	    // error. Here, we attach tag with conduitid, as each conduit has unque id,
		// the mpi msg will have different new tag
		MPI_Send(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

		// the first thread finish real write, change finishflag
		LCK1.lock();
#ifdef USE_DEBUG_ASSERT
		assert(m_WriteOpRotateFinishFlag[counteridx] == 0);
#endif
		// set finish flag
		m_WriteOpRotateFinishFlag[counteridx]++;
		// update counter idx to next one
		m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
		// notify other late coming threads to exit
		m_WriteOpFinishCond.notify_all();
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish write:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish write:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

		return 0;

	}


	return 0;
}

int XprocConduit::WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" call writeby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(myThreadRank >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cerr<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}
		// not the writing thread, just return, we will not wait for the real write
		// finish!
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit writeby!"<<std::endl;
#endif
		return 0;
	}

	// current thread is the writing thread
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing writeby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

	// doing real data write
#ifdef USE_MPI_BASE
	MPI_Send(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

	// record this op to readby finish set
	std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
	m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
	m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish writeby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish writeby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

	return 0;
}

void XprocConduit::WriteBy_Finish(int tag)
{

#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    static thread_local int myTaskid = -1;
    static thread_local int myThreadRank = -1;
    if(myTaskid == -1)
    {
        myTaskid = TaskManager::getCurrentTaskId();
        myThreadRank = TaskManager::getCurrentThreadRankinTask();
    }

    std::unique_lock<std::mutex> LCK1(m_writebyFinishMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" wait for writeby..."<<std::endl;
#endif
	while(m_writebyFinishSet.find((tag<<LOG_MAX_TASKS)+myTaskid) ==
			m_writebyFinishSet.end())
	{
		// tag not in finishset, so not finish yet
		m_writebyFinishCond.wait(LCK1);
	}
	// find tag in finishset
#ifdef USE_DEBUG_ASSERT
	assert(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
#endif
	m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
	if(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
	{
		m_writebyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
	}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" finish wait writeby!"<<std::endl;
#endif

	return;
}



////////////////
int XprocConduit::BWrite(void* DataPtr, int DataSize, int tag)
{
	std::cerr<<"Error, crossing process conduit doen't has 'BWrite' method."<<std::endl;
	return 0;
}
int XprocConduit::BWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
{
	std::cerr<<"Error, crossing process conduit doen't has 'BWriteBy' method."<<std::endl;
	return 0;
}
void XprocConduit::BWriteBy_Finish(int tag)
{
	std::cerr<<"Error, crossing process conduit doen't has 'BWriteBy_Finish' method."<<std::endl;
	return;
}



////////////////
int XprocConduit::PWrite(void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}


	std::unique_lock<std::mutex> LCK1(m_WriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" call Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif
	int counteridx = m_WriteOpRotateCounterIdx[myThreadRank];
	m_WriteOpRotateCounter[counteridx]++;
	if(m_WriteOpRotateCounter[counteridx]>1)
	{
		// a late thread
#ifdef USE_DEBUG_ASSERT
		assert(m_WriteOpRotateCounter[counteridx] <= localNumthreads);
#endif
		while(m_WriteOpRotateFinishFlag[counteridx] == 0)
		{
			m_WriteOpFinishCond.wait(LCK1);
		}
		// wake up
		m_WriteOpRotateFinishFlag[counteridx]++;
		if(m_WriteOpRotateFinishFlag[counteridx]==localNumthreads)
		{
			//last leaving thread
			m_WriteOpRotateFinishFlag[counteridx] = 0;
			m_WriteOpRotateCounter[counteridx] = 0;
			//
			m_availableNoFinishedWriteOpCount++;
			m_availableNoFinishedWriteCond.notify_one();

			// update counter idx to next one
			m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
			LCK1.unlock();
		}
		else
		{
			// update counter idx to next one
			m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
			LCK1.unlock();
		}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit Pwrite:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

		return 0;
	}
	else
	{
		// first thread
		while(m_availableNoFinishedWriteOpCount == 0)
		{
			m_availableNoFinishedWriteCond.wait(LCK1);
		}
		m_availableNoFinishedWriteOpCount--;
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		MPI_Ssend(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

		// the first thread finish real write, change finishflag
		LCK1.lock();
#ifdef USE_DEBUG_ASSERT
		assert(m_WriteOpRotateFinishFlag[counteridx] == 0);
#endif
		// set finish flag
		m_WriteOpRotateFinishFlag[counteridx]++;
		// update counter idx to next one
		m_WriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
		// notify other late coming threads to exit
		m_WriteOpFinishCond.notify_all();
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

		return 0;

	}

	return 0;
}

int XprocConduit::PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" call Pwriteby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(myThreadRank >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cerr<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}
		// not the writing thread, just return, we will not wait for the real write
		// finish!
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit Pwriteby!"<<std::endl;
#endif
		return 0;
	}

	// current thread is the writing thread
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwriteby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwriteby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

	// doing real data write
#ifdef USE_MPI_BASE
	MPI_Ssend(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

	// record this op to readby finish set
	std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
	m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
	m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwriteby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwriteby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
#endif

	return 0;
}

void XprocConduit::PWriteBy_Finish(int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    static thread_local int myTaskid = -1;
    static thread_local int myThreadRank = -1;
    if(myTaskid == -1)
    {
        myTaskid = TaskManager::getCurrentTaskId();
        myThreadRank = TaskManager::getCurrentThreadRankinTask();
    }

    std::unique_lock<std::mutex> LCK1(m_writebyFinishMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" wait for Pwriteby..."<<std::endl;
#endif
	while(m_writebyFinishSet.find((tag<<LOG_MAX_TASKS)+myTaskid) ==
			m_writebyFinishSet.end())
	{
		// tag not in finishset, so not finish yet
		m_writebyFinishCond.wait(LCK1);
	}
	// find tag in finishset
#ifdef USE_DEBUG_ASSERT
	assert(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
#endif
	m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
	if(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
	{
		m_writebyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
	}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" finish wait Pwriteby!"<<std::endl;
#endif

	return;
}


int XprocConduit::Read(void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif

	std::unique_lock<std::mutex> LCK1(m_ReadOpCheckMutex);
	int counteridx = m_ReadOpRotateCounterIdx[myThreadRank];
	m_ReadOpRotateCounter[counteridx]++;
	if(m_ReadOpRotateCounter[counteridx]>1)
	{
		// late coming thread
#ifdef USE_DEBUG_ASSERT
		assert(m_ReadOpRotateCounter[counteridx] <= localNumthreads);
#endif
		while(m_ReadOpRotateFinishFlag[counteridx] ==0)
		{
			m_ReadOpFinishCond.wait(LCK1);
		}
		// wake up after real read finish
		m_ReadOpRotateFinishFlag[counteridx]++;
		if(m_ReadOpRotateFinishFlag[counteridx] == localNumthreads)
		{
			//last leaving thread
			m_ReadOpRotateFinishFlag[counteridx]=0;
			m_ReadOpRotateCounter[counteridx]=0;
			m_ReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);

			m_availableNoFinishedReadOpCount++;
			m_availableNoFinishedReadCond.notify_one();

			LCK1.unlock();
		}
		else
		{
			m_ReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);
			LCK1.unlock();
		}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif
		return 0;
	}
	else
	{
		// first coming thread
		while(m_availableNoFinishedReadOpCount=0)
		{
			m_availableNoFinishedReadCond.wait(LCK1);
		}
		m_availableNoFinishedReadOpCount--;
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		MPI_Recv(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif

		// change readfinishflag
		LCK1.lock();
#ifdef USE_DEBUG_ASSERT
		assert(m_ReadOpRotateFinishFlag[counteridx] ==0);
#endif
		m_ReadOpRotateFinishFlag[counteridx]++;
		m_ReadOpFinishCond.notify_all();
		m_ReadOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_capacity+1);
		LCK1.unlock();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif

		return 0;
	}

	return 0;
}

int XprocConduit::ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	static thread_local int mpiOtherEndProc = -1;
	static thread_local int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(mpiOtherEndProc == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		if(myTaskid == m_srcId)
		{
			mpiOtherEndProc = m_dstMainResideProc;
			localNumthreads = m_numSrcLocalThreads;
		}
		else if(myTaskid == m_dstId)
		{
			mpiOtherEndProc = m_srcMainResideProc;
			localNumthreads = m_numDstLocalThreads;
		}
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" call readby..."<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(myThreadRank >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cerr<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}
		// not the writing thread, just return, we will not wait for the real write
		// finish!
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit readby!"<<std::endl;
#endif
		return 0;
	}

	// current thread is the designated thread
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
	MPI_Recv(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
			(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif

	// record this op in finish set
	std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
	m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
	m_readbyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish readby:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish readby:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
#endif

	return 0;
}

void XprocConduit::ReadBy_Finish(int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    static thread_local int myTaskid = -1;
    static thread_local int myThreadRank = -1;
    if(myTaskid == -1)
    {
        myTaskid = TaskManager::getCurrentTaskId();
        myThreadRank = TaskManager::getCurrentThreadRankinTask();
    }

    std::unique_lock<std::mutex> LCK1(m_readbyFinishMutex);
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" wait for readby..."<<std::endl;
#endif
    while(m_readbyFinishSet.find((tag<<LOG_MAX_TASKS)+myTaskid) ==
            m_readbyFinishSet.end())
    {
        // tag not in finishset, so not finish yet
        m_readbyFinishCond.wait(LCK1);
    }
    // find tag in finishset
#ifdef USE_DEBUG_ASSERT
    assert(m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
#endif
    m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
    if(m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
    {
        // actually we need not modify tag at here, as right now this finishset is only
    	// used by src or dst, they are in different process, both src and dst will
    	// have a conduit object.
    	// so, all data member in cdt obj that named with read-  and named with write-
    	// only one of them is used, as src/dst do not share one cdt obj like inproc-conduit
    	// does.
        m_readbyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
    }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" finish wait readby!"<<std::endl;
#endif


	return;
}

XprocConduit::~XprocConduit()
{
	delete m_WriteOpRotateCounter;
	delete m_WriteOpRotateCounterIdx;
	delete m_WriteOpRotateFinishFlag;

	delete m_ReadOpRotateCounter;
	delete m_ReadOpRotateCounterIdx;
	delete m_ReadOpRotateFinishFlag;

	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();

#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"XprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
#endif

}


}//end namespace iUtc
