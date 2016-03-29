#include "XprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "../include/TaskUtilities.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>


namespace iUtc{

thread_local std::ofstream *XprocConduit::m_threadOstream = nullptr;

XprocConduit::XprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId, std::string name)
:ConduitBase()
{
	//
	m_conduitId = cdtId;
	m_srcTask = srctask;
	m_dstTask = dsttask;
	m_Name = name;
	m_srcId = m_srcTask->getTaskId();
	m_dstId = m_dstTask->getTaskId();

	m_numSrcLocalThreads = srctask->getNumLocalThreads();
	m_numDstLocalThreads = dsttask->getNumLocalThreads();

	m_srcMainResideProc = srctask->getMainResideProcess();
	m_dstMainResideProc = dsttask->getMainResideProcess();

	//
	int numLocalThreads = m_numSrcLocalThreads > m_numDstLocalThreads?
			m_numSrcLocalThreads: m_numDstLocalThreads;
	m_OpTokenFlag = new int[numLocalThreads];
	/*m_OpThreadAtomic = new std::atomic<int>[numLocalThreads];
	for(int i=0; i<numLocalThreads; i++){
		m_OpTokenFlag[i]=0;
		boost::latch *tmp_latch = new boost::latch(1);
		m_OpThreadLatch.push_back(tmp_latch);
		m_OpThreadAtomic[i].store(1);
	}*/
	for(int i=0; i<numLocalThreads; i++)
		m_OpTokenFlag[i]=0;
	m_OpThreadAvailable.push_back(new std::atomic<int>(0));
	m_OpThreadFinish.push_back(new std::atomic<intptr_t>((intptr_t)new boost::latch(1)));


	//
#ifdef ENABLE_OPBY_FINISH
	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();
#endif


	//
#ifdef USE_MPI_BASE
	m_asyncReadFinishSet.clear();
	m_asyncWriteFinishSet.clear();
#endif
	m_asyncOpTokenFlag = new int[numLocalThreads];
	/*m_asyncOpThreadAtomic = new std::atomic<int>[numLocalThreads];
	for(int i=0; i<numLocalThreads;i++){
		m_asyncOpTokenFlag[i] = 0;
		m_asyncOpThreadAtomic[i].store(1);
	}*/
	for(int i=0; i<numLocalThreads; i++)
		m_asyncOpTokenFlag[i]=0;
	m_asyncOpThreadAvailable.push_back(new std::atomic<int>(0));
	m_asyncOpThreadFinish.push_back(new std::atomic<int>(1));


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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	static thread_local int myLocalRank = -1;
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
	if(localNumthreads==1){
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
		MPI_Datatype datatype=MPI_CHAR;
		if(DataSize > ((unsigned)1<<31)-1){
			DataSize = (DataSize+3)/4;
			datatype = MPI_INT;
		}
		MPI_Send(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
		//*m_threadOstream<<"here "<<mpiOtherEndProc<<" "<<(tag<<LOG_MAX_CONDUITS)+m_conduitId<<std::endl;
#endif
	}// end one thread
	else
	{	//multiple threads
		int idx = m_OpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_OpThreadAvailable.size());
			//assert(m_OpThreadFinish[idx]->load() != 0);
#endif
		if(!m_OpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			// a late thread
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
				if(!temp_latch->try_wait()){
					temp_latch->wait();
				}
			}
			else{
				long _counter=0;
				while(m_OpThreadFinish[idx]->load() !=0){
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
			}
			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue= m_OpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){
					delete m_OpThreadAvailable[idx];
					m_OpThreadAvailable[idx] = nullptr;
					boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
					if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
						delete temp_latch;
					delete m_OpThreadFinish[idx];
					m_OpThreadFinish[idx]=nullptr;
					break;
				}
				if(m_OpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
					break;
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
		}
		else{
			// the first coming thread
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_OpThreadAvailable.size()-1);
#endif
			// push next item to vector
			m_OpThreadAvailable.push_back(new std::atomic<int>(0));
			boost::latch* tmp_latch= new boost::latch(1);
			m_OpThreadFinish.push_back(new std::atomic<intptr_t>((intptr_t)tmp_latch));

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
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Send(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

			// wake up other threads
			tmp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				tmp_latch->count_down();
			}
			else{
				delete tmp_latch;
				m_OpThreadFinish[idx]->store(0);
			}
		}
		m_OpTokenFlag[myLocalRank]++;

	}// end multi threads
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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
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
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"thread "<<myThreadRank<<" call writeby..."<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
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
	MPI_Datatype datatype=MPI_CHAR;
	if(DataSize > ((unsigned)1<<31)-1){
		DataSize = (DataSize+3)/4;
		datatype = MPI_INT;
	}
	MPI_Send(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

#ifdef ENABLE_OPBY_FINISH
	// record this op to readby finish set
	if(localNumthreads>1)
	{
		std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
		m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_writebyFinishCond.notify_all();
	}
#endif

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

#ifdef ENABLE_OPBY_FINISH
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
#endif



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
#ifdef ENABLE_OPBY_FINISH
void XprocConduit::BWriteBy_Finish(int tag)
{
	std::cerr<<"Error, crossing process conduit doen't has 'BWriteBy_Finish' method."<<std::endl;
	return;
}
#endif



////////////////
int XprocConduit::PWrite(void* DataPtr, DataSize_t DataSize, int tag)
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

	if(localNumthreads == 1){
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
		MPI_Datatype datatype=MPI_CHAR;
		if(DataSize > ((unsigned)1<<31)-1){
			DataSize = (DataSize+3)/4;
			datatype = MPI_INT;
		}
		MPI_Ssend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif
	}// end one thread
	else
	{	// there are several threads
		int idx = m_OpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_OpThreadAvailable.size());
			//assert(m_OpThreadFinish[idx]->load() != 0);
#endif
		if(!m_OpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			// a late thread
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
				if(!temp_latch->try_wait()){
					temp_latch->wait();
				}
			}
			else{
				long _counter=0;
				while(m_OpThreadFinish[idx]->load() !=0){
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
			}
			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue= m_OpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){
					delete m_OpThreadAvailable[idx];
					m_OpThreadAvailable[idx] = nullptr;
					boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
					if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
						delete temp_latch;
					delete m_OpThreadFinish[idx];
					m_OpThreadFinish[idx]=nullptr;
					break;
				}
				if(m_OpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
					break;
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

		}
		else{
			// this thread's turn to do op
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_OpThreadAvailable.size()-1);
#endif
			// push next item to vector
			m_OpThreadAvailable.push_back(new std::atomic<int>(0));
			boost::latch* tmp_latch= new boost::latch(1);
			m_OpThreadFinish.push_back(new std::atomic<intptr_t>((intptr_t)tmp_latch));

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
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Ssend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif
			tmp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				tmp_latch->count_down();
			}
			else{
				delete tmp_latch;
				m_OpThreadFinish[idx]->store(0);
			}
		}
		m_OpTokenFlag[myLocalRank]++;
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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
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
	*m_threadOstream<<"thread "<<myThreadRank<<" call Pwriteby..."<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
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
	MPI_Datatype datatype=MPI_CHAR;
	if(DataSize > ((unsigned)1<<31)-1){
		DataSize = (DataSize+3)/4;
		datatype = MPI_INT;
	}
	MPI_Ssend(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD);
#endif

#ifdef ENABLE_OPBY_FINISH
	if(localNumthreads >1)
	{
		// record this op to readby finish set
		std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
		m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_writebyFinishCond.notify_all();
	}
#endif

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

#ifdef ENABLE_OPBY_FINISH
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
#endif


int XprocConduit::Read(void* DataPtr, DataSize_t DataSize, int tag)
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
		MPI_Datatype datatype=MPI_CHAR;
		if(DataSize > ((unsigned)1<<31)-1){
			DataSize = (DataSize+3)/4;
			datatype = MPI_INT;
		}
		MPI_Recv(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
				(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
	} // end only one thread
	else
	{	// there are several threads
		int idx = m_OpTokenFlag[myLocalRank];
		int isavailable =0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_OpThreadAvailable.size());
			//assert(m_OpThreadFinish[idx]->load() != 0);
#endif
		if(!m_OpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
			// a late thread
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
				if(!temp_latch->try_wait()){
					temp_latch->wait();
				}
			}
			else{
				long _counter=0;
				while(m_OpThreadFinish[idx]->load() !=0){
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
			}
			int nthreads = localNumthreads-1;
			while(1){
				int oldvalue= m_OpThreadAvailable[idx]->load();
				if(oldvalue == nthreads){
					delete m_OpThreadAvailable[idx];
					m_OpThreadAvailable[idx] = nullptr;
					boost::latch *temp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
					if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
						delete temp_latch;
					delete m_OpThreadFinish[idx];
					m_OpThreadFinish[idx]=nullptr;
					break;
				}
				if(m_OpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
					break;
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
		}
		else
		{
			// this thread's turn to do r/w
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_OpThreadAvailable.size()-1);
#endif
			// push next item to vector
			m_OpThreadAvailable.push_back(new std::atomic<int>(0));
			boost::latch* tmp_latch= new boost::latch(1);
			m_OpThreadFinish.push_back(new std::atomic<intptr_t>((intptr_t)tmp_latch));

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
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Recv(DataPtr, (int)DataSize, datatype, mpiOtherEndProc,
					(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
			tmp_latch = (boost::latch*)m_OpThreadFinish[idx]->load();
			if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
				tmp_latch->count_down();
			}
			else{
				delete tmp_latch;
				m_OpThreadFinish[idx]->store(0);
			}

		}
		m_OpTokenFlag[myLocalRank]++;
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
	int mpiOtherEndProc = -1;
	int localNumthreads = -1;
	static thread_local int myThreadRank = -1;
	static thread_local int myTaskid=-1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
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
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" call readby..."<<std::endl;
#endif

	if(myThreadRank != thread)
	{
		if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
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
	//TODO:
	MPI_Datatype datatype=MPI_CHAR;
	if(DataSize > ((unsigned)1<<31)-1){
		DataSize = (DataSize+3)/4;
		datatype = MPI_INT;
	}
	//*m_threadOstream<<"here "<<mpiOtherEndProc<<" "<<(tag<<LOG_MAX_CONDUITS)+m_conduitId<<std::endl;
	MPI_Recv(DataPtr, DataSize, datatype, mpiOtherEndProc,
			(tag<<LOG_MAX_CONDUITS)+m_conduitId, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
	//*m_threadOstream<<"here "<<mpiOtherEndProc<<std::endl;
#ifdef ENABLE_OPBY_FINISH
	if(localNumthreads>1)
	{
		// record this op in finish set
		std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
		m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = localNumthreads;
		m_readbyFinishCond.notify_all();
	}
#endif

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

#ifdef ENABLE_OPBY_FINISH
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
#endif

XprocConduit::~XprocConduit()
{
	//
	free(m_OpTokenFlag);
	/*for(auto &it: m_OpThreadLatch){
		delete it;
	}
	m_OpThreadLatch.clear();
	delete m_OpThreadAtomic;*/
	m_OpThreadAvailable.clear();
	m_OpThreadFinish.clear();

	//
#ifdef ENABLE_OPBY_FINISH
	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();
#endif

	//
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
	delete m_asyncOpTokenFlag;
	/*delete m_asyncOpThreadAtomic;*/
	m_asyncOpThreadAvailable.clear();
	m_asyncOpThreadFinish.clear();


#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"XprocConduit: ["<<m_Name<<"] destroyed !!!"<<std::endl;
#endif

}


}//end namespace iUtc
