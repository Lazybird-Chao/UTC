#include "XprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include <cassert>
#include "../include/TaskUtilities.h"

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
		MPI_Datatype datatype=MPI_CHAR;
		if(DataSize > (1<<31)-1){
			DataSize = (DataSize+3)/4;
			datatype = MPI_INT;
		}
		MPI_Irecv(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
		m_asyncReadFinishSet[tag] = req;

#endif

	}// end only one thread
	else
	{	// multple threads
		if(myThreadRank == m_asyncOpTokenFlag[myThreadRank]){
			//
			int next_thread = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;
			m_asyncOpThreadAtomic[next_thread].store(0);

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncRead..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > (1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Irecv(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncReadFinishSet[tag] = req;
#endif
			m_asyncOpThreadAtomic[m_asyncOpTokenFlag[myThreadRank]].store(1);
			m_asyncOpTokenFlag[myThreadRank] = next_thread;

		}
		else
		{	// not this thread's turn to do
			int do_thread = m_asyncOpTokenFlag[myThreadRank];
			while(m_asyncOpThreadAtomic[do_thread].load() == 0){
				_mm_pause();
			}
			//
			m_asyncOpTokenFlag[myThreadRank] = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;

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
		if(myThreadRank!= m_asyncOpTokenFlag[myThreadRank]){
			int do_thread = m_asyncOpTokenFlag[myThreadRank];
			while(m_asyncOpThreadAtomic[do_thread].load() == 0){
				_mm_pause();
			}
			//
			m_asyncOpTokenFlag[myThreadRank] = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;
		}
		else
		{
			// this thread's turn
			int next_thread = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;
			m_asyncOpThreadAtomic[next_thread].store(0);

#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncReadFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncReadFinishSet.erase(tag);
#endif
			m_asyncOpThreadAtomic[m_asyncOpTokenFlag[myThreadRank]].store(1);
			m_asyncOpTokenFlag[myThreadRank]= next_thread;

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
		MPI_Datatype datatype=MPI_CHAR;
		if(DataSize > (1<<31)-1){
			DataSize = (DataSize+3)/4;
			datatype = MPI_INT;
		}
		MPI_Isend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
		m_asyncWriteFinishSet[tag] = req;

#endif

	}// end only one thread
	else
	{	// multiple threads
		if(myThreadRank == m_asyncOpTokenFlag[myThreadRank]){
			//
			int next_thread = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;
			m_asyncOpThreadAtomic[next_thread].store(0);

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncWrite..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > (1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Isend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncWriteFinishSet[tag] = req;
#endif
			m_asyncOpThreadAtomic[m_asyncOpTokenFlag[myThreadRank]].store(1);
			m_asyncOpTokenFlag[myThreadRank]=next_thread;

		}
		else
		{	// do wait
			int do_thread = m_asyncOpTokenFlag[myThreadRank];
			while(m_asyncOpThreadAtomic[do_thread].load() == 0){
				_mm_pause();
			}
			//
			m_asyncOpTokenFlag[myThreadRank] = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;

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
		if(m_asyncOpTokenFlag[myThreadRank] != myThreadRank){
			// do wait
			int do_thread = m_asyncOpTokenFlag[myThreadRank];
			while(m_asyncOpThreadAtomic[do_thread].load() == 0){
				_mm_pause();
			}
			//
			m_asyncOpTokenFlag[myThreadRank] = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;

		}
		else
		{
			int next_thread = (m_asyncOpTokenFlag[myThreadRank]+1)%localNumthreads;
			m_asyncOpThreadAtomic[next_thread].store(0);

#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncWriteFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncWriteFinishSet.erase(tag);
#endif
			m_asyncOpThreadAtomic[m_asyncOpTokenFlag[myThreadRank]].store(1);
			m_asyncOpTokenFlag[myThreadRank]=next_thread;

		}// end first thread
	}// end several thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
	return;
}




}// end namespace iUtc
