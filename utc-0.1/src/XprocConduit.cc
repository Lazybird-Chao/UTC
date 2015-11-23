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

	m_noFinishedOpCapacity = NO_FINISHED_OP_MAX;

	m_availableNoFinishedOpCount = m_noFinishedOpCapacity;
	m_OpRotateCounter = new int[m_noFinishedOpCapacity+1];
	m_OpRotateFinishFlag = new int[m_noFinishedOpCapacity+1];
	for(int i =0;i<m_noFinishedOpCapacity+1; i++)
	{
		m_OpRotateCounter[i]=0;
		m_OpRotateFinishFlag[i]=0;
	}
	//actually only use the local threads in one process, other thread pos is not used
	int numLocalthreads = m_srcTask->getNumTotalThreads() > m_dstTask->getNumTotalThreads()?
			m_srcTask->getNumTotalThreads():m_dstTask->getNumTotalThreads();
	m_OpRotateCounterIdx = new int[numLocalthreads];
	for(int i =0; i<numLocalthreads;i++)
	{
		m_OpRotateCounterIdx[i] = 0;
	}

	//
	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();


	//
#ifdef USE_MPI_BASE
	m_asyncReadFinishSet.clear();
	m_asyncWriteFinishSet.clear();
#endif

#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"XprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] initiated..."<<std::endl;
#endif
}

int XprocConduit::Write(void* DataPtr, DataSize_t DataSize, int tag)
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

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" call write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif
	if(localNumthreads==1)
	{
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		// TODO: solving int(datasize) problem
		MPI_Send(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif
	}
	else
	{
		std::unique_lock<std::mutex> LCK1(m_OpCheckMutex);
		int counteridx = m_OpRotateCounterIdx[myThreadRank];
		m_OpRotateCounter[counteridx]++;
		if(m_OpRotateCounter[counteridx]>1)
		{
			// a late thread
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateCounter[counteridx] <= localNumthreads);
#endif
			while(m_OpRotateFinishFlag[counteridx] == 0)
			{
				m_OpFinishCond.wait(LCK1);
			}
			// wake up
			m_OpRotateFinishFlag[counteridx]++;
			if(m_OpRotateFinishFlag[counteridx]==localNumthreads)
			{
				//last leaving thread
				m_OpRotateFinishFlag[counteridx] = 0;
				m_OpRotateCounter[counteridx] = 0;
				//
				m_availableNoFinishedOpCount++;
				m_availableNoFinishedCond.notify_one();

				// update counter idx to next one
				m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
				LCK1.unlock();
			}
			else
			{
				// update counter idx to next one
				m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit write:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

			return 0;
		}
		else
		{
			// first thread
			while(m_availableNoFinishedOpCount == 0)
			{
				m_availableNoFinishedCond.wait(LCK1);
			}
			m_availableNoFinishedOpCount--;
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
			// because in one proc, there may be several tasks, so two different mpi msg
			// could have same src/dst proc, if these two msg send by different tasks use
			// same tag, then mpi msg would has same msg-envelop, may cause msg matching
			// error. Here, we attach tag with conduitid, as each conduit has unque id,
			// the mpi msg will have different new tag
			// TODO: solving int(datasize) problem
			MPI_Send(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

			// the first thread finish real write, change finishflag
			LCK1.lock();
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateFinishFlag[counteridx] == 0);
#endif
			// set finish flag
			m_OpRotateFinishFlag[counteridx]++;
			// update counter idx to next one
			m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
			// notify other late coming threads to exit
			m_OpFinishCond.notify_all();
			LCK1.unlock();
		}

	}
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish write:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish write:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

	return 0;
}

int XprocConduit::WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" call writeby..."<<std::endl;
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

	// doing real data write
#ifdef USE_MPI_BASE
	// TODO: int-datasize limitation
	MPI_Send(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

	// record this op to readby finish set
	if(localNumthreads>1)
	{
		std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
		m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_writebyFinishCond.notify_all();
	}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish writeby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish writeby:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
    if(myTaskid == m_srcId && m_numSrcLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit wait writeby!"<<std::endl;
#endif
		return;
	}
	else if(myTaskid == m_dstId && m_numDstLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit wait writeby!"<<std::endl;
#endif
		return;
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
int XprocConduit::BWrite(void* DataPtr, DataSize_t DataSize, int tag)
{
	std::cerr<<"Error, crossing process conduit doen't has 'BWrite' method."<<std::endl;
	return 0;
}
int XprocConduit::BWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
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
int XprocConduit::PWrite(void* DataPtr, DataSize_t DataSize, int tag)
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
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" call Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

	if(localNumthreads == 1)
	{
#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		// TODO: int-datasize limitation
		MPI_Ssend(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif
	}
	else
	{	// there are several threads
		std::unique_lock<std::mutex> LCK1(m_OpCheckMutex);
		int counteridx = m_OpRotateCounterIdx[myThreadRank];
		m_OpRotateCounter[counteridx]++;
		if(m_OpRotateCounter[counteridx]>1)
		{
			// a late thread
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateCounter[counteridx] <= localNumthreads);
#endif
			while(m_OpRotateFinishFlag[counteridx] == 0)
			{
				m_OpFinishCond.wait(LCK1);
			}
			// wake up
			m_OpRotateFinishFlag[counteridx]++;
			if(m_OpRotateFinishFlag[counteridx]==localNumthreads)
			{
				//last leaving thread
				m_OpRotateFinishFlag[counteridx] = 0;
				m_OpRotateCounter[counteridx] = 0;
				//
				m_availableNoFinishedOpCount++;
				m_availableNoFinishedCond.notify_one();

				// update counter idx to next one
				m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
				LCK1.unlock();
			}
			else
			{
				// update counter idx to next one
				m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

			return 0;
		}
		else
		{
			// first thread
			while(m_availableNoFinishedOpCount == 0)
			{
				m_availableNoFinishedCond.wait(LCK1);
			}
			m_availableNoFinishedOpCount--;
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
			// TODO: int-datasize limitation
			MPI_Ssend(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

			// the first thread finish real write, change finishflag
			LCK1.lock();
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateFinishFlag[counteridx] == 0);
#endif
			// set finish flag
			m_OpRotateFinishFlag[counteridx]++;
			// update counter idx to next one
			m_OpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
			// notify other late coming threads to exit
			m_OpFinishCond.notify_all();
			LCK1.unlock();
		}// end for first thread

	}// end for several threads

#ifdef USE_DEBUG_LOG
		if(myTaskid == m_srcId)
		{
			PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
		}
		else
		{
			PRINT_TIME_NOW(*m_threadOstream)
				*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
		}
#endif

	return 0;
}

int XprocConduit::PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" call Pwriteby..."<<std::endl;
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwriteby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
	}
#endif

	// doing real data write
#ifdef USE_MPI_BASE
	// TODO: int-datasize limitation
	MPI_Ssend(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

	if(localNumthreads >1)
	{
		// record this op to readby finish set
		std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
		m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_writebyFinishCond.notify_all();
	}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwriteby:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwriteby:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
    if(myTaskid == m_srcId && m_numSrcLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit wait Pwriteby!"<<std::endl;
#endif
		return;
	}
	else if(myTaskid == m_dstId && m_numDstLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit wait Pwriteby!"<<std::endl;
#endif
		return;
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


int XprocConduit::Read(void* DataPtr, DataSize_t DataSize, int tag)
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	}
#endif

	if(localNumthreads == 1)
	{

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
		// TODO:
		MPI_Recv(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
	} // end only one thread
	else
	{	// there are several threads
		std::unique_lock<std::mutex> LCK1(m_OpCheckMutex);
		int counteridx = m_OpRotateCounterIdx[myThreadRank];
		m_OpRotateCounter[counteridx]++;
		if(m_OpRotateCounter[counteridx]>1)
		{
			// late coming thread
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateCounter[counteridx] <= localNumthreads);
#endif
			while(m_OpRotateFinishFlag[counteridx] ==0)
			{
				m_OpFinishCond.wait(LCK1);
			}
			// wake up after real read finish
			m_OpRotateFinishFlag[counteridx]++;
			if(m_OpRotateFinishFlag[counteridx] == localNumthreads)
			{
				//last leaving thread
				m_OpRotateFinishFlag[counteridx]=0;
				m_OpRotateCounter[counteridx]=0;
				m_OpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);

				m_availableNoFinishedOpCount++;
				m_availableNoFinishedCond.notify_one();

				LCK1.unlock();
			}
			else
			{
				m_OpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	}
#endif
			return 0;
		}
		else
		{
			// first coming thread
			while(m_availableNoFinishedOpCount==0)
			{
				m_availableNoFinishedCond.wait(LCK1);
			}
			m_availableNoFinishedOpCount--;
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
			// TODO:
			MPI_Recv(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif

			// change readfinishflag
			LCK1.lock();
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateFinishFlag[counteridx] ==0);
#endif
			m_OpRotateFinishFlag[counteridx]++;
			m_OpFinishCond.notify_all();
			m_OpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
			LCK1.unlock();

		}// end first thread
	}// end several threads
#ifdef USE_DEBUG_LOG
		if(myTaskid == m_srcId)
		{
			PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
		}
		else
		{
			PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
		}
#endif
	return 0;
}

int XprocConduit::ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
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
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	}
#endif

#ifdef USE_MPI_BASE
	MPI_Recv(DataPtr, DataSize, MPI_CHAR, mpiOtherEndProc,
			(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif

	if(localNumthreads>1)
	{
		// record this op in finish set
		std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
		m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_readbyFinishCond.notify_all();
	}

#ifdef USE_DEBUG_LOG
	if(myTaskid == m_srcId)
	{
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish readby:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	}
	else
	{
		PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish readby:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
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
    if(myTaskid == m_srcId && m_numSrcLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait readby!"<<std::endl;
#endif
		return;
	}
	else if(myTaskid == m_dstId && m_numDstLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait readby!"<<std::endl;
#endif
		return;
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
	delete m_OpRotateCounter;
	delete m_OpRotateCounterIdx;
	delete m_OpRotateFinishFlag;

	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();

#ifdef USE_MPI_BASE
	for(auto &it:m_asyncReadFinishSet)
	{
		free(it.second);
	}
	m_asyncReadFinishSet.clear();
	for(auto &it:m_asyncWriteFinishSet)
	{
		free(it.second);
	}
	m_asyncWriteFinishSet.clear();

#endif


#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"XprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
#endif

}


}//end namespace iUtc
