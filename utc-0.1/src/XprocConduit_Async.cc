#include "XprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "Task_Utilities.h"

#include <cassert>

namespace iUtc{

int XprocConduit::AsyncRead(void* DataPtr, DataSize_t DataSize, int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" call AsyncRead..."<<std::endl;
#endif

	if(localNumthreads == 1)
	{

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncRead..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
		MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
		//TODO:
		MPI_Irecv(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
		m_asyncReadFinishSet[tag] = req;

#endif

	}// end only one thread
	else
	{
		std::unique_lock<std::mutex> LCK1(m_OpCheckMutex);
		int counteridx = m_OpRotateCounterIdx[myThreadRank];
		m_OpRotateCounter[counteridx]++;
		if(m_OpRotateCounter[counteridx] == 1)
		{
			// fisrt coming thread
			while(m_availableNoFinishedOpCount==0)
			{
				m_availableNoFinishedCond.wait(LCK1);
			}
			m_availableNoFinishedOpCount--;

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncRead..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Irecv(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncReadFinishSet[tag] = req;
#endif
			m_OpRotateFinishFlag[counteridx]++;
			m_OpFinishCond.notify_all();
			m_OpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
			LCK1.unlock();

		}
		else
		{	// late coming thread
			while(m_OpRotateFinishFlag[counteridx] ==0)
			{
				m_OpFinishCond.wait(LCK1);
			}
			m_OpRotateFinishFlag[counteridx]++;
			if(m_OpRotateFinishFlag[counteridx] == localNumthreads)
			{
				// last thread
				m_OpRotateCounter[counteridx]=0;
				m_OpRotateFinishFlag[counteridx]=0;
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
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" exit AsyncRead..."<<std::endl;
#endif
			return 0;

		}

	}// end several threads
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish AsyncRead..."<<std::endl;
#endif
	return 0;
}

void XprocConduit::AsyncRead_Finish(int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" wait for AsyncRead..."<<std::endl;
#endif

	if(localNumthreads ==1)
	{
#ifdef USE_MPI_BASE
		MPI_Request* req = m_asyncReadFinishSet[tag];
		MPI_Wait(req, MPI_STATUS_IGNORE);
		free(req);
		m_asyncReadFinishSet.erase(tag);
#endif
	}
	else
	{
		// there are several threads
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
			// wake up after real op finish
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

#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncReadFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncReadFinishSet.erase(tag);
#endif

			// change opfinishflag
			LCK1.lock();
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateFinishFlag[counteridx] ==0);
#endif
			m_OpRotateFinishFlag[counteridx]++;
			m_OpFinishCond.notify_all();
			m_OpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
			LCK1.unlock();

		}// end first thread
	}// end several thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish wait AsyncRead!"<<std::endl;
#endif
	return;
}


int XprocConduit::AsyncWrite(void *DataPtr, DataSize_t DataSize, int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" call AsyncWrite..."<<std::endl;
#endif

	if(localNumthreads == 1)
	{

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncWrite..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
		MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
		//TODO:
		MPI_Isend(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
		m_asyncWriteFinishSet[tag] = req;

#endif

	}// end only one thread
	else
	{
		std::unique_lock<std::mutex> LCK1(m_OpCheckMutex);
		int counteridx = m_OpRotateCounterIdx[myThreadRank];
		m_OpRotateCounter[counteridx]++;
		if(m_OpRotateCounter[counteridx] == 1)
		{
			// fisrt coming thread
			while(m_availableNoFinishedOpCount==0)
			{
				m_availableNoFinishedCond.wait(LCK1);
			}
			m_availableNoFinishedOpCount--;

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncWrite..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Isend(DataPtr, (int)DataSize, MPI_CHAR, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncWriteFinishSet[tag] = req;
#endif
			m_OpRotateFinishFlag[counteridx]++;
			m_OpFinishCond.notify_all();
			m_OpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
			LCK1.unlock();

		}
		else
		{	// late coming thread
			while(m_OpRotateFinishFlag[counteridx] ==0)
			{
				m_OpFinishCond.wait(LCK1);
			}
			m_OpRotateFinishFlag[counteridx]++;
			if(m_OpRotateFinishFlag[counteridx] == localNumthreads)
			{
				// last thread
				m_OpRotateCounter[counteridx]=0;
				m_OpRotateFinishFlag[counteridx]=0;
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
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" exit AsyncWrite..."<<std::endl;
#endif
			return 0;

		}

	}// end several threads
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish AsyncWrite..."<<std::endl;
#endif
	return 0;

}


void XprocConduit::AsyncWrite_Finish(int tag)
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
	*m_threadOstream<<"thread "<<myThreadRank<<" wait for AsyncWrite..."<<std::endl;
#endif

	if(localNumthreads ==1)
	{
#ifdef USE_MPI_BASE
		MPI_Request* req = m_asyncWriteFinishSet[tag];
		MPI_Wait(req, MPI_STATUS_IGNORE);
		free(req);
		m_asyncWriteFinishSet.erase(tag);
#endif
	}
	else
	{
		// there are several threads
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
			// wake up after real op finish
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

#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncWriteFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncWriteFinishSet.erase(tag);
#endif

			// change opfinishflag
			LCK1.lock();
#ifdef USE_DEBUG_ASSERT
			assert(m_OpRotateFinishFlag[counteridx] ==0);
#endif
			m_OpRotateFinishFlag[counteridx]++;
			m_OpFinishCond.notify_all();
			m_OpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
			LCK1.unlock();

		}// end first thread
	}// end several thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
	return;
}




}// end namespace iUtc
