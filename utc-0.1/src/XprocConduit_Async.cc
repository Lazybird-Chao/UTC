#include "XprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "../include/TaskUtilities.h"
#include <cassert>


namespace iUtc{

int XprocConduit::AsyncRead(void* DataPtr, DataSize_t DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	static thread_local int myLocalRank =-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
		myLocalRank = TaskManager::getCurrentThreadRankInLocal();
	}
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
	else
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
		if(DataSize > ((unsigned)1<<31)-1){
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
		int idx = m_asyncOpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_asyncOpThreadAvailable.size());
			//assert(m_asyncOpThreadFinish[idx]->load() != 0);
#endif
		if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			//

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncRead..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Irecv(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncReadFinishSet[tag] = req;
#endif
			// wake up other threads
			m_asyncOpThreadFinish[idx]->store(0);

		}
		else
		{	// not this thread's turn to do
			long _counter=0;
			while(m_asyncOpThreadFinish[idx]->load() != 0){
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
			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue =m_asyncOpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){

					m_asyncOpThreadAvailable[idx]->store(0);

					m_asyncOpThreadFinish[idx]->store(1);
					break;
				}
				if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
					break;
			}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" exit AsyncRead..."<<std::endl;
#endif
		}
		m_asyncOpTokenFlag[myLocalRank]++;
		m_asyncOpTokenFlag[myLocalRank]=m_asyncOpTokenFlag[myLocalRank]%m_nOps2;

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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	static thread_local int myLocalRank=-1;
	if(myTaskid== -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
		myLocalRank = TaskManager::getCurrentThreadRankInLocal();
	}
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
	else
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
		int idx = m_asyncOpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_asyncOpThreadAvailable.size());
			//assert(m_asyncOpThreadFinish[idx]->load() != 0);
#endif
		if(!m_asyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			// a late thread
			long _counter=0;
			while(m_asyncOpThreadFinish[idx]->load() !=0){
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

			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue =m_asyncOpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){

					m_asyncOpThreadAvailable[idx]->store(0);

					m_asyncOpThreadFinish[idx]->store(1);
					break;
				}
				if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
					break;
			}
		}
		else
		{
			// this thread's turn

#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncReadFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncReadFinishSet.erase(tag);
#endif
			m_asyncOpThreadFinish[idx]->store(0);

		}// end first thread
		m_asyncOpTokenFlag[myLocalRank]++;
		m_asyncOpTokenFlag[myLocalRank]=m_asyncOpTokenFlag[myLocalRank]%m_nOps2;
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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	static thread_local int myLocalRank=-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
		myLocalRank = TaskManager::getCurrentThreadRankInLocal();
	}
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
	else
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
		if(DataSize > ((unsigned)1<<31)-1){
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
		int idx = m_asyncOpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_asyncOpThreadAvailable.size());
			//assert(m_asyncOpThreadFinish[idx]->load() != 0);
#endif
		if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			//

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" doing AsyncWrite..."<<std::endl;
#endif

#ifdef USE_MPI_BASE
			MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request));
			//TODO:
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Isend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, req);
			m_asyncWriteFinishSet[tag] = req;
#endif
			m_asyncOpThreadFinish[idx]->store(0);

		}
		else
		{	// do wait
			// a late thread
			long _counter=0;
			while(m_asyncOpThreadFinish[idx]->load() !=0){
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

			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue =m_asyncOpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){

					m_asyncOpThreadAvailable[idx]->store(0);

					m_asyncOpThreadFinish[idx]->store(1);
					break;
				}
				if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
					break;
			}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" exit AsyncWrite..."<<std::endl;
#endif

		}
		//
		m_asyncOpTokenFlag[myLocalRank]++;
		m_asyncOpTokenFlag[myLocalRank]=m_asyncOpTokenFlag[myLocalRank]%m_nOps2;

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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	static thread_local int myLocalRank =-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
		myLocalRank = TaskManager::getCurrentThreadRankInLocal();
	}
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
	else
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
		int idx = m_asyncOpTokenFlag[myLocalRank];
		int isavailable =0;
		if(!m_asyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			// do wait
			// a late thread
			long _counter=0;
			while(m_asyncOpThreadFinish[idx]->load() !=0){
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

			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue =m_asyncOpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){

					m_asyncOpThreadAvailable[idx]->store(0);

					m_asyncOpThreadFinish[idx]->store(1);
					break;
				}
				if(m_asyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
					break;
			}
		}
		else
		{


#ifdef USE_MPI_BASE
			MPI_Request* req = m_asyncWriteFinishSet[tag];
			MPI_Wait(req, MPI_STATUS_IGNORE);
			free(req);
			m_asyncWriteFinishSet.erase(tag);
#endif
			m_asyncOpThreadFinish[idx]->store(0);

		}// end first thread
		m_asyncOpTokenFlag[myLocalRank]++;
		m_asyncOpTokenFlag[myLocalRank]=m_asyncOpTokenFlag[myLocalRank]%m_nOps2;
	}// end several thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
	return;
}




}// end namespace iUtc
