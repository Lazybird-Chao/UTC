#include "Conduit.h"
#include "ConduitManager.h"
#include "TaskManager.h"
#include "Task.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>

namespace iUtc
{

thread_local std::ofstream *Conduit::m_threadOstream = nullptr;

Conduit::Conduit()
{
    m_srcTask = nullptr;
    m_dstTask = nullptr;
    m_srcId = -1;
    m_dstId = -1;

    m_numSrcLocalThreads = 0;
    m_numDstLocalThreads = 0;
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
#ifdef USE_DEBUG_LOG
    std::ofstream *procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: [dummy conduit] constructed..."<<std::endl;
#endif
    //initConduit();   call init through connect()

}
void Conduit::checkOnSameProc(TaskBase* src, TaskBase* dst)
{
	if((src->isActiveOnCurrentProcess()== false && dst->isActiveOnCurrentProcess()== true) ||
			(src->isActiveOnCurrentProcess()== true && dst->isActiveOnCurrentProcess()==false))
	{
		std::cout<<"Error, two Tasks are not running on same process!"<<std::endl;
		exit(1);
	}
}
Conduit::Conduit(TaskBase* srctask, TaskBase* dsttask)
{
	checkOnSameProc(srctask, dsttask);
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] constructed..."<<std::endl;
#endif
    initConduit();

}

Conduit::Conduit(TaskBase* srctask, TaskBase* dsttask, int capacity)
{
	checkOnSameProc(srctask,dsttask);
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();
    if(capacity > CONDUIT_CAPACITY_MAX)
    {
    	m_capacity = CONDUIT_CAPACITY_MAX;
    }
    else
    	m_capacity = capacity;
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] constructed..."<<std::endl;
#endif
    initConduit();

}


void Conduit::initConduit()
{
	if(m_srcTask->isActiveOnCurrentProcess()== false && m_dstTask->isActiveOnCurrentProcess()== false)
	{	// no thread for src and dst running on this process, no need to init the conduit obj
		return;
	}
	m_Name = m_srcTask->getName()+"<=>"+m_dstTask->getName();
	m_conduitId = ConduitManager::getNewConduitId();


	m_srcAvailableBuffCount = m_capacity;
	m_srcBuffPool.clear();
	m_srcBuffIdx.clear();
	//m_srcBuffAccessMutex.clear();
	std::vector<std::mutex> *tmp1_mutexlist = new std::vector<std::mutex>(m_capacity);
	m_srcBuffAccessMutex.swap(*tmp1_mutexlist);
	std::vector<std::condition_variable> *tmp1_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_srcBuffWcallbyAllCond.swap(*tmp1_condlist);
	m_srcBuffWrittenFlag.clear();
	for(int i = 0; i< m_capacity; i++)
	{
		m_srcBuffIdx.push_back(i);
		m_srcBuffWrittenFlag.push_back(0);
	}
	m_srcWriteOpRotateCounter = new int[m_capacity+1];
	m_srcReadOpRotateCounter = new int[m_capacity +1];
	m_srcWriteOpRotateFinishFlag = new int[m_capacity+1];
	m_srcReadOpRotateFinishFlag = new int[m_capacity+1];
	for(int i =0;i<m_capacity+1; i++)
	{
		m_srcWriteOpRotateCounter[i]=0;
		m_srcReadOpRotateCounter[i]=0;
		m_srcWriteOpRotateFinishFlag[i]=0;
		m_srcReadOpRotateFinishFlag[i]=0;
	}
	m_srcWriteOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	//actually only use the local threads in one process, other thread pos is not used
	m_srcReadOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	for(int i =0; i<m_srcTask->getNumTotalThreads();i++)
	{
		m_srcWriteOpRotateCounterIdx[i] = 0;
		m_srcReadOpRotateCounterIdx[i]=0;
	}

	m_dstAvailableBuffCount = m_capacity;
	m_dstAvailableBuffCount = m_capacity;
	m_dstBuffPool.clear();
	m_dstBuffIdx.clear();
	//m_dstBuffAccessMutex.clear();
	std::vector<std::mutex> *tmp2_mutexlist= new std::vector<std::mutex>(m_capacity);
	m_dstBuffAccessMutex.swap(*tmp2_mutexlist);
	std::vector<std::condition_variable> *tmp2_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_dstBuffWcallbyAllCond.swap(*tmp2_condlist);
	m_dstBuffWrittenFlag.clear();
	for(int i = 0; i< m_capacity; i++)
	{
		m_dstBuffIdx.push_back(i);
		m_dstBuffWrittenFlag.push_back(0);
	}
	m_dstWriteOpRotateCounter = new int[m_capacity+1];
	m_dstReadOpRotateCounter = new int[m_capacity +1];
	m_dstWriteOpRotateFinishFlag = new int[m_capacity+1];
	m_dstReadOpRotateFinishFlag = new int[m_capacity+1];
	for(int i =0;i<m_capacity+1; i++)
	{
		m_dstWriteOpRotateCounter[i]=0;
		m_dstReadOpRotateCounter[i]=0;
		m_dstWriteOpRotateFinishFlag[i]=0;
		m_dstReadOpRotateFinishFlag[i]=0;
	}
	m_dstWriteOpRotateCounterIdx = new int[m_dstTask->getNumTotalThreads()];
	m_dstReadOpRotateCounterIdx = new int[m_dstTask->getNumTotalThreads()];
	for(int i =0; i<m_dstTask->getNumTotalThreads();i++)
	{
		m_dstWriteOpRotateCounterIdx[i] = 0;
		m_dstReadOpRotateCounterIdx[i]=0;
	}


	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();


	ConduitManager* cdtMgr = ConduitManager::getInstance();
	m_cdtMgr = cdtMgr;
	cdtMgr->registerConduit(this, m_conduitId);


#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] initiated..."<<std::endl;
#endif

}


Conduit::~Conduit()
{
    if(m_srcTask)
    {
        if(TaskManager::hasTaskItem(m_srcId))
        {   // task not destroyed
            if(m_srcTask->hasActiveLocalThread())
            {
                // there are task threads still running
                m_srcTask->waitLocalThreadFinish();
            }
        }
    }
    if(m_dstTask)
    {
        if(TaskManager::hasTaskItem(m_dstId))
            if(m_dstTask->hasActiveLocalThread())
                m_dstTask->waitLocalThreadFinish();
    }
    // delete this conduit item from conduit registry
	m_cdtMgr->unregisterConduit(this, m_conduitId);

	clear();

}

void Conduit::clear()
{

	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();

    m_srcBuffIdx.clear();
    m_srcBuffAccessMutex.clear();
    m_srcBuffWcallbyAllCond.clear();
    m_srcBuffWrittenFlag.clear();
    delete m_srcWriteOpRotateCounter;
    delete m_srcReadOpRotateCounter;
    delete m_srcWriteOpRotateCounterIdx;
    delete m_srcReadOpRotateCounterIdx;
    delete m_srcWriteOpRotateFinishFlag;
    delete m_srcReadOpRotateFinishFlag;

    m_dstBuffIdx.clear();
    m_dstBuffAccessMutex.clear();
    m_dstBuffWcallbyAllCond.clear();
    m_dstBuffWrittenFlag.clear();
    delete m_dstWriteOpRotateCounter;
    delete m_dstReadOpRotateCounter;
    delete m_dstWriteOpRotateCounterIdx;
    delete m_dstReadOpRotateCounterIdx;
    delete m_dstWriteOpRotateFinishFlag;
    delete m_dstReadOpRotateFinishFlag;

    for(std::map<MessageTag, BuffInfo*>::iterator it = m_srcBuffPool.begin();
            it != m_srcBuffPool.end(); ++it)
    {
        if((it->second)->dataPtr)
            free((it->second)->dataPtr);
        delete it->second;
    }
    m_srcBuffPool.clear();
    for(std::map<MessageTag, BuffInfo*>::iterator it = m_dstBuffPool.begin();
            it != m_dstBuffPool.end(); ++it)
    {
        if((it->second)->dataPtr)
            free((it->second)->dataPtr);
        delete it->second;
    }
    m_dstBuffPool.clear();
#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    if(m_srcTask && m_dstTask)
    {
        *procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
    }
    else
    {
        *procOstream<<"Conduit: [dummy conduit] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
    }
#endif
}

// can't change capacity after conduit is created
/*void Conduit::setCapacity(int capacity)
{
    m_capacity = capacity;
}*/

int Conduit::getCapacity()
{
    return m_capacity;
}

std::string Conduit::getName()
{
	return m_Name;
}

TaskBase* Conduit::getSrcTask()
{
    return m_srcTask;
}

TaskBase* Conduit::getDstTask()
{
    return m_dstTask;
}

TaskBase* Conduit::getAnotherTask()
{
	static thread_local int myTaskid = -1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
	}
	if(myTaskid == m_srcId)
		return m_dstTask;
	else if(myTaskid == m_dstId)
		return m_srcTask;
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

}

void Conduit::Connect(TaskBase* src, TaskBase* dst)
{
    if(m_srcTask || m_dstTask)
    {
        std::cout<<"Error, already connected to some Task"<<std::endl;
        exit(1);
    }
    checkOnSameProc(src, dst);
    m_srcTask = src;
    m_dstTask = dst;
    m_srcId = src->getTaskId();
    m_dstId = dst->getTaskId();
    m_numSrcLocalThreads = src->getNumLocalThreads();
    m_numDstLocalThreads = dst->getNumLocalThreads();
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] connected..."<<std::endl;
#endif
    initConduit();
    return;
}

ConduitId Conduit::getConduitId()
{
    return m_conduitId;
}


int Conduit::Write(void *DataPtr, int DataSize, int tag)
{
#ifdef USE_DEBUG_LOG
	if(!m_threadOstream)
		m_threadOstream = getThreadOstream();
#endif
	// current calling thread's belonging task id
	static thread_local int myTaskid = -1;
	static thread_local int myThreadRank = -1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
		myThreadRank = TaskManager::getCurrentThreadRankinTask();
	}

	if(myTaskid == m_srcId)
	{
		// srctask calling write()

		// get write op lock and check op counter value to see if need do real write
		std::unique_lock<std::mutex> LCK1(m_srcWriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		int counteridx = m_srcWriteOpRotateCounterIdx[myThreadRank];
		m_srcWriteOpRotateCounter[counteridx]++;
		if(m_srcWriteOpRotateCounter[counteridx] >1)
		{
			// a late coming thread, but at most = all local threads
			assert(m_srcWriteOpRotateCounter[counteridx] <= m_numSrcLocalThreads);

			while(m_srcWriteOpRotateFinishFlag[counteridx] == 0)
			{
				// the first thread which do real write hasn't finish, we will wait for it to finish
				m_srcWriteOpFinishCond.wait(LCK1);
			}
			// wake up, so the write is finished
			m_srcWriteOpRotateFinishFlag[counteridx]++;
			if(m_srcWriteOpRotateFinishFlag[counteridx] == m_numSrcLocalThreads)
			{
				// last thread that will leave this write, reset counter and flag value
				m_srcWriteOpRotateFinishFlag[counteridx] = 0;
				m_srcWriteOpRotateCounter[counteridx] = 0;
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();

				// last thread update some info related with the buff inserted to buffpool
				std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);  		///////////////// TODO: may use buffer's access mutex
				assert(m_srcBuffPool.find(tag) != m_srcBuffPool.end());
				assert(m_srcBuffPool[tag]->callingWriteThreadCount == 1);  //only the real write thread has modified this value
				assert(m_srcBuffPool[tag]->safeReleaseAfterRead == false);
				m_srcBuffPool[tag]->callingWriteThreadCount = m_numSrcLocalThreads;
				m_srcBuffPool[tag]->safeReleaseAfterRead = true;
				// in case there is reader thread waiting for this to release buff
				m_srcBuffWcallbyAllCond[m_srcBuffPool[tag]->buffIdx].notify_one();
				LCK2.unlock();
			}
			else
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();
			}
			return 0;

		}
		else
		{
			LCK1.unlock();
			// the first coming thread, who will do real write
			std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);

			// check if there is available buff
			while(m_srcAvailableBuffCount == 0)
			{
				// buffpool full, wait for one
				m_srcBuffAvailableCond.wait(LCK2);
			}
			// get buff, go on check if tag exist in pool
			if(m_srcBuffPool.find(tag) != m_srcBuffPool.end())
			{
				// exist, this would be an tag reuse error
				std::cout<<"Error, tag resued!"<<std::endl;
				LCK2.unlock();
				exit(1);
			}
			else
			{
				// has buff and tag not exist now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// allocate space
				tmp_buffinfo->dataPtr = malloc(DataSize);
				// get buff id
				tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
				m_srcBuffIdx.pop_back();
				// set count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				if(m_numSrcLocalThreads == 1)
				{
					// only one local thread, set this flag here, no chance going to late coming thread process
					tmp_buffinfo->safeReleaseAfterRead = true;
					// no need to call wcallbyallcond.notify, as no reader thread's release can happen before this time point
				}
				// insert this buff to buffpool
				m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_srcAvailableBuffCount--;
				// get access lock for this buffer to write data
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				// release buffmanager lock to allow other threads to get buff
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_srcNewBuffInsertedCond.notify_all();
				// do real data transfer
				memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_srcBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// release access lock
				LCK3.unlock();
				// notify reader to read data. (may not need, as writer hold the access lock
				// during transfer, so even reader find the tag in pool, he will block at getting
				// access lock.
				/*m_srcBuffAccessCond[tmp_buffinfo->buffId].nitify_all(); */

				// the first thread finish real write, change finishflag
				LCK1.lock();
				assert(m_srcWriteOpRotateFinishFlag[counteridx] == 0);
				// set finish flag
				m_srcWriteOpRotateFinishFlag[counteridx]++;
				if(m_numSrcLocalThreads == 1)
				{	// only one local thread, reset counter and flag at here
					m_srcWriteOpRotateFinishFlag[counteridx] = 0;
					m_srcWriteOpRotateCounter[counteridx] = 0;
				}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish write!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				// notify other late coming threads to exit
				m_srcWriteOpFinishCond.notify_all();
				LCK1.unlock();
				return 0;
			}// end do real write
		}// end first coming thread

	}//end srcTask
	else if(myTaskid == m_dstId)
	{
		// dsttask calling write()

		// get write op lock and check op counter value to see if need do real write
		std::unique_lock<std::mutex> LCK1(m_dstWriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		int counteridx = m_dstWriteOpRotateCounterIdx[myThreadRank];
		m_dstWriteOpRotateCounter[counteridx]++;
		if(m_dstWriteOpRotateCounter[counteridx] >1)
		{
			// a late coming thread, but at most = all local threads
			assert(m_dstWriteOpRotateCounter[counteridx] <= m_numDstLocalThreads);

			while(m_dstWriteOpRotateFinishFlag[counteridx] == 0)
			{
				// the first thread which do real write hasn't finish, we will wait for it to finish
				m_dstWriteOpFinishCond.wait(LCK1);
			}
			// wake up, so the write is finished
			m_dstWriteOpRotateFinishFlag[counteridx]++;
			if(m_dstWriteOpRotateFinishFlag[counteridx] == m_numDstLocalThreads)
			{
				// last thread that will leave this write, reset counter and flag value
				m_dstWriteOpRotateFinishFlag[counteridx] = 0;
				m_dstWriteOpRotateCounter[counteridx] = 0;
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();

				// last thread update some info related with the buff inserted to buffpool
				std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
				////////// TODO: may use access mutex, then no need to race this manager mutex, but reader side who will check safe release flag
				////////// and do release may has problem, as he need managermutex to release a buffer
				assert(m_dstBuffPool.find(tag) != m_dstBuffPool.end());
				assert(m_dstBuffPool[tag]->callingWriteThreadCount == 1);
				assert(m_dstBuffPool[tag]->safeReleaseAfterRead == false);
				m_dstBuffPool[tag]->callingWriteThreadCount = m_numDstLocalThreads;
				m_dstBuffPool[tag]->safeReleaseAfterRead = true;
				// in case there is reader thread waiting for this to release buff
				m_dstBuffWcallbyAllCond[m_dstBuffPool[tag]->buffIdx].notify_one();
				LCK2.unlock();
			}
			else
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();
			}
			return 0;

		}
		else
		{
			LCK1.unlock();
			// the first coming thread, who will do real write
			std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);

			// check if there is available buff
			while(m_dstAvailableBuffCount == 0)
			{
				// buffpool full, wait for one
				m_dstBuffAvailableCond.wait(LCK2);
			}
			// get buff, go on check if tag exist in pool
			if(m_dstBuffPool.find(tag) != m_dstBuffPool.end())
			{
				// exist, this would be an tag reuse error
				std::cout<<"Error, tag resued!"<<std::endl;
				LCK2.unlock();
				exit(1);
			}
			else
			{
				// has buff now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// allocate space
				tmp_buffinfo->dataPtr = malloc(DataSize);
				// get buff id
				tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
				m_dstBuffIdx.pop_back();
				// set count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				if(m_numDstLocalThreads == 1)
				{	// only one local thread, set this flag here, no chance going to late coming thread process
					tmp_buffinfo->safeReleaseAfterRead = true;
				}
				// insert this buff to buffpool
				m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_dstAvailableBuffCount--;
				// get access lock for this buffer to write data
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				// release buffmanager lock to allow other threads to get buff
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_dstNewBuffInsertedCond.notify_all();
				// do real data transfer
				memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_dstBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// release access lock
				LCK3.unlock();
				// notify reader to read data. (may not need, as writer hold the access lock
				// during transfer, so even reader find the tag in pool, he will block at getting
				// access lock.
				/*m_dstBuffAccessCond[tmp_buffinfo->buffId].nitify_all(); */

				// the first thread finish real write
				LCK1.lock();
				assert(m_dstWriteOpRotateFinishFlag[counteridx] == 0);
				// set finish flag
				m_dstWriteOpRotateFinishFlag[counteridx]++;
				if(m_numSrcLocalThreads == 1)
				{	// only one local thread, reset counter and flag at here
					m_dstWriteOpRotateFinishFlag[counteridx] = 0;
					m_dstWriteOpRotateCounter[counteridx] = 0;
				}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish write!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				// notify other late coming threads to exit
				m_dstWriteOpFinishCond.notify_all();
				LCK1.unlock();
				return 0;
			}// end do real write
		}// end first coming thread

	}//end dstTask
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}
}// end Write()


int Conduit::Read(void *DataPtr, int DataSize, int tag)
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
		//*m_threadOstream<<"dst-here"<<myTaskid<<std::endl;
	}

	if(myTaskid == m_srcId)
	{
		// src task calling read()

		// get read op lock and check if need do real read
		std::unique_lock<std::mutex> LCK1(m_srcReadOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		int counteridx = m_srcReadOpRotateCounterIdx[myThreadRank];
		m_srcReadOpRotateCounter[counteridx]++;
		if(m_srcReadOpRotateCounter[counteridx] > 1)
		{
			// a late coming thread
			assert(m_srcReadOpRotateCounter[counteridx] <= m_numSrcLocalThreads);
			while(m_srcReadOpRotateFinishFlag[counteridx] ==0)
			{
				m_srcReadOpFinishCond.wait(LCK1);
			}
			// wake up after real read finish
			m_srcReadOpRotateFinishFlag[counteridx]++;
			if(m_srcReadOpRotateFinishFlag[counteridx] == m_numSrcLocalThreads)
			{
				// last read thread that is leaving
				m_srcReadOpRotateCounter[counteridx]=0;
				m_srcReadOpRotateFinishFlag[counteridx] = 0;
				// update counter idx to next one
				m_srcReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);
				// avoid prohibiting other threads go on their reading while this thread is waiting
				LCK1.unlock();

				std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
				assert(m_dstBuffPool.find(tag) != m_dstBuffPool.end());
				assert(m_dstBuffPool[tag]->callingReadThreadCount == 1);
				// last read thread will release the buff
				BuffInfo *tmp_buff = m_dstBuffPool[tag];
				while(tmp_buff->safeReleaseAfterRead == false)
				{
					m_dstBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);

				}
				assert(tmp_buff->dataPtr!=nullptr);
				free(tmp_buff->dataPtr);
				tmp_buff->dataPtr = nullptr;
				m_dstBuffWrittenFlag[tmp_buff->buffIdx] =0;
				m_dstBuffIdx.push_back(tmp_buff->buffIdx);
				delete tmp_buff;
				m_dstAvailableBuffCount++;
				m_dstBuffPool.erase(tag);
				// a buffer released, notify writer there are available buff now
				m_dstBuffAvailableCond.notify_one();
				LCK2.unlock();
			}
			else
			{
				// update counter idx to next one
				m_srcReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);
				LCK1.unlock();
			}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			return 0;
		}// end late coming read thread
		else
		{
			// the first coming thread, do real read
			LCK1.unlock();
			std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
			std::cv_status w_ret = std::cv_status::no_timeout;
			while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
			{
				// no tag in buffpool, means the message hasn't come yet
				// wait for writer insert the message to conduit buffpool
				w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
				if(w_ret == std::cv_status::timeout)
				{
					std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
					LCK2.unlock();
					exit(1);
				}
			}
			// find the tag in the pool, the msg has come, but data may not finish transfer
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			BuffInfo* tmp_buff = m_dstBuffPool[tag];
			assert(m_dstAvailableBuffCount < m_capacity);
			assert(m_dstBuffPool[tag]->callingReadThreadCount == 0);
			m_dstBuffPool[tag]->callingReadThreadCount = 1;
			// get access lock, will block for waiting writer transferring
			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buff->buffIdx]);
			// release buff manager lock to allow other thread doing read or write
			LCK2.unlock();
			assert(m_dstBuffWrittenFlag[tmp_buff->buffIdx] == 1);
			// real data transfer
			memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
			LCK3.unlock();

			// the first thread finish real read need change the readfinishflag
			LCK1.lock();
			assert(m_srcReadOpRotateFinishFlag[counteridx] ==0);
			if(m_numSrcLocalThreads ==1)
			{
				// only one local thread read
				m_srcReadOpRotateCounter[counteridx] = 0;
				//m_srcReadOpRotateFinishFlag[counteridx] = 0;
				m_srcReadOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_capacity+1);
				LCK1.unlock();

				//need release buff after read
				LCK2.lock();
				while(tmp_buff->safeReleaseAfterRead == false)
				{
					m_dstBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);
				}
				assert(tmp_buff->dataPtr!=nullptr);
				free(tmp_buff->dataPtr);
				tmp_buff->dataPtr = nullptr;
				m_dstBuffWrittenFlag[tmp_buff->buffIdx] =0;
				m_dstBuffIdx.push_back(tmp_buff->buffIdx);
				delete tmp_buff;
				m_dstAvailableBuffCount++;
				m_dstBuffPool.erase(tag);
				// a buffer released, notify writer there are available buff now
				m_dstBuffAvailableCond.notify_one();
				LCK2.unlock();

			}
			else
			{
				m_srcReadOpRotateFinishFlag[counteridx]++;
				m_srcReadOpFinishCond.notify_all();
				m_srcReadOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_capacity+1);
				LCK1.unlock();
			}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif

			return 0;
		}// end real read

	}// end src read
	else if(myTaskid == m_dstId)
	{
		// dst task calling read()

		// get read op lock and check if need do real read
		std::unique_lock<std::mutex> LCK1(m_dstReadOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
		int counteridx = m_dstReadOpRotateCounterIdx[myThreadRank];
		m_dstReadOpRotateCounter[counteridx]++;
		if(m_dstReadOpRotateCounter[counteridx] > 1)
		{
			// a late coming thread
			assert(m_dstReadOpRotateCounter[counteridx] <= m_numDstLocalThreads);
			while(m_dstReadOpRotateFinishFlag[counteridx] ==0)
			{
				m_dstReadOpFinishCond.wait(LCK1);
			}
			// wake up after real read finish
			m_dstReadOpRotateFinishFlag[counteridx]++;
			if(m_dstReadOpRotateFinishFlag[counteridx] == m_numDstLocalThreads)
			{
				// last read thread that is leaving
				m_dstReadOpRotateCounter[counteridx]=0;
				m_dstReadOpRotateFinishFlag[counteridx] = 0;
				// update counter idx to next one
				m_dstReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);
				LCK1.unlock();

				// also need do the buff release
				std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
				assert(m_srcBuffPool.find(tag) != m_srcBuffPool.end());
				assert(m_srcBuffPool[tag]->callingReadThreadCount == 1);
				// last read thread will release the buff
				BuffInfo *tmp_buff = m_srcBuffPool[tag];
				while(tmp_buff->safeReleaseAfterRead == false)
				{
					m_srcBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);

				}
				assert(tmp_buff->dataPtr!=nullptr);
				free(tmp_buff->dataPtr);
				tmp_buff->dataPtr = nullptr;
				m_srcBuffWrittenFlag[tmp_buff->buffIdx] =0;
				m_srcBuffIdx.push_back(tmp_buff->buffIdx);
				delete tmp_buff;
				m_srcAvailableBuffCount++;
				m_srcBuffPool.erase(tag);
				// a buffer released, notify writer there are available buff now
				m_srcBuffAvailableCond.notify_one();
				LCK2.unlock();
			}
			else
			{
				// update counter idx to next one
				m_dstReadOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_capacity +1);
				LCK1.unlock();
			}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			return 0;
		}// end late coming read thread
		else
		{
			// the first coming thread, do real read
			LCK1.unlock();
			std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
			std::cv_status w_ret = std::cv_status::no_timeout;
			while(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
			{
				// no tag in buffpool, means the message hasn't come yet
				// wait for writer insert the message to conduit buffpool
				w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
				if(w_ret == std::cv_status::timeout)
				{
					std::cout<<"Error, reader wait time out!"<<"dst-thread "<<myThreadRank<<std::endl;
					LCK2.unlock();
					exit(1);
				}
			}
			// find the tag in the pool, the msg has come, but data may not finish transfer
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			BuffInfo* tmp_buff = m_srcBuffPool[tag];
			assert(m_srcAvailableBuffCount < m_capacity);
			assert(m_srcBuffPool[tag]->callingReadThreadCount == 0);
			m_srcBuffPool[tag]->callingReadThreadCount = 1;
			// get access lock, will block for waiting writer transferring
			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buff->buffIdx]);
			// release buff manager lock to allow other thread doing read or write
			LCK2.unlock();
			assert(m_srcBuffWrittenFlag[tmp_buff->buffIdx] == 1);
			// real data transfer
			memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
			LCK3.unlock();

			// the first thread finish real read need change the readfinishflag
			LCK1.lock();
			assert(m_dstReadOpRotateFinishFlag[counteridx] ==0);
			if(m_numDstLocalThreads ==1)
			{
				// only one local thread read
				m_dstReadOpRotateCounter[counteridx] = 0;
				//m_dstReadOpRotateFinishFlag[counteridx] = 0;
				m_dstReadOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_capacity+1);
				LCK1.unlock();

				// need release buff after read
				LCK2.lock();
				while(tmp_buff->safeReleaseAfterRead == false)
				{
					m_srcBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);
				}
				assert(tmp_buff->dataPtr!=nullptr);
				free(tmp_buff->dataPtr);
				tmp_buff->dataPtr = nullptr;
				m_srcBuffWrittenFlag[tmp_buff->buffIdx] =0;
				m_srcBuffIdx.push_back(tmp_buff->buffIdx);
				delete tmp_buff;
				m_srcAvailableBuffCount++;
				m_srcBuffPool.erase(tag);
				// a buffer released, notify writer there are available buff now
				m_srcBuffAvailableCond.notify_one();
				LCK2.unlock();

			}
			else
			{
				m_dstReadOpRotateFinishFlag[counteridx]++;
				m_dstReadOpFinishCond.notify_all();
				m_dstReadOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_capacity+1);
				LCK1.unlock();
			}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			return 0;

		}// end real read
	}// end dst read
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// end read


int Conduit::WriteBy(ThreadRank thread, void *DataPtr, int DataSize, int tag)
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
		//*m_threadOstream<<"dst-here"<<myTaskid<<std::endl;
	}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" call writeby..."<<std::endl;
#endif
	if(myThreadRank != thread)
	{
		if(myThreadRank >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cout<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
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
	if(myTaskid == m_srcId)
	{
		//srctask calling write()
		// As only the assigned task do write, so no need to use op lock and op counter
		// to prevent other threads from doing this write op.

		// For writeby, we use same buffpool as write, so we can pair writeby/read and
		// write/readby, write/read, writby/readby freely.

		// Check if there's buff available
		std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
		m_srcBuffAvailableCond.wait(LCK2,
				[=](){return m_srcAvailableBuffCount != 0;});
		// get buff, go on checking if tag reuse
		if(m_srcBuffPool.find(tag) != m_srcBuffPool.end())
		{
			// tag already exist
			std::cout<<"Error, tag resued!"<<std::endl;
			LCK2.unlock();
			exit(1);
		}
		else
		{
			// has buff and tag not exist, do real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing writeby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			BuffInfo* tmp_buffinfo = new BuffInfo;
			// allocate space
			tmp_buffinfo->dataPtr = malloc(DataSize);
			// get buff id
			tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
			m_srcBuffIdx.pop_back();
			// set count to 1
			tmp_buffinfo->callingWriteThreadCount = 1;
			// only one thread do write, set safe release flag
			tmp_buffinfo->safeReleaseAfterRead = true;
			// insert this buff to buffpool
			m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_srcAvailableBuffCount--;
			// get access lock for this buffer to write data
			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
			// release buffmanager lock to allow other threads to get buff
			LCK2.unlock();
			// notify reader that one new item inserted to buff pool
			m_srcNewBuffInsertedCond.notify_all();
			// do real data transfer
			memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
			m_srcBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
			// release access lock
			LCK3.unlock();

			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
			//		m_numSrcLocalThreads-1));
			m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
			m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish writeby!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			return 0;
		}

	}
	else if(myTaskid == m_dstId)
	{
		//dsttask calling write
		std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
		m_dstBuffAvailableCond.wait(LCK2,
				[=](){return m_dstAvailableBuffCount !=0;});
		// get buff, go on check if tag exist in pool
		if(m_dstBuffPool.find(tag) != m_dstBuffPool.end())
		{
			// exist, this would be an tag reuse error
			std::cout<<"Error, tag resued!"<<std::endl;
			LCK2.unlock();
			exit(1);
		}
		else
		{
			// has buff now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			BuffInfo* tmp_buffinfo = new BuffInfo;
			// allocate space
			tmp_buffinfo->dataPtr = malloc(DataSize);
			// get buff id
			tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
			m_dstBuffIdx.pop_back();
			// set count to 1
			tmp_buffinfo->callingWriteThreadCount = 1;
			// set safe release flag
			tmp_buffinfo->safeReleaseAfterRead = true;
			// insert this buff to buffpool
			m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_dstAvailableBuffCount--;
			// get access lock for this buffer to write data
			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
			// release buffmanager lock to allow other threads to get buff
			LCK2.unlock();
			// notify reader that one new item inserted to buff pool
			m_dstNewBuffInsertedCond.notify_all();
			// do real data transfer
			memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
			m_dstBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
			// release access lock
			LCK3.unlock();

			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
			//		m_numDstLocalThreads-1));
			m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
			m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish writeby!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			return 0;
		}
	}
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// end WriteBy()

void Conduit::WriteBy_Finish(int tag)
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
		// we use (tag<<LOG_MAX_TASKS)+myTaskid as the internal tag here,
		// as for src and dst we use one finish set, so to need differentiate
		// the srctag and dsttag

		// tag not in finishset, so not finish yet
		m_writebyFinishCond.wait(LCK1);
	}
	// find tag in finishset
	assert(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
	m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
	if(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
	{
		//last thread do the finish check, erase this tag from finish set
		// as we erase this tag, so after all threads call this function,
		// another call with same tag value may cause infinite waiting.
		// In one thread, do not call finish check with same tag for two or more
		// times, unless write a new msg with this tag.
		m_writebyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
	}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" finish wait writeby!"<<std::endl;
#endif

	return;

}// end WriteBy_Finish


int Conduit::ReadBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
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
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" call readby..."<<std::endl;
#endif
	if(myThreadRank != thread)
	{
		if(myThreadRank >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cout<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit readby!"<<std::endl;
#endif
		return 0;
	}

	// assigned thread do the real read
	if(myTaskid == m_srcId)
	{
		std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
		std::cv_status w_ret = std::cv_status::no_timeout;
		while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
		{
			// no tag in buffpool, means the message hasn't come yet
			// wait for writer insert the message to conduit buffpool
			w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
			if(w_ret == std::cv_status::timeout)
			{
				std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
				LCK2.unlock();
				exit(1);
			}
		}

		// find the tag in the pool, the msg has come, but data may not finish transfer
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		BuffInfo* tmp_buff = m_dstBuffPool[tag];
		assert(m_dstAvailableBuffCount < m_capacity);
		assert(m_dstBuffPool[tag]->callingReadThreadCount == 0);
		m_dstBuffPool[tag]->callingReadThreadCount = 1;
		// get access lock, will block for waiting writer transferring
		std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buff->buffIdx]);
		// release buff manager lock to allow other thread doing read or write
		LCK2.unlock();
		assert(m_dstBuffWrittenFlag[tmp_buff->buffIdx] == 1);
		// real data transfer
		memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
		LCK3.unlock();

		// as only this assigned thread do read, so release buff after read
		LCK2.lock();
		while(tmp_buff->safeReleaseAfterRead == false)
		{
			m_dstBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);
		}
		assert(tmp_buff->dataPtr!=nullptr);
		free(tmp_buff->dataPtr);
		tmp_buff->dataPtr = nullptr;
		m_dstBuffWrittenFlag[tmp_buff->buffIdx] =0;
		m_dstBuffIdx.push_back(tmp_buff->buffIdx);
		delete tmp_buff;
		m_dstAvailableBuffCount++;
		m_dstBuffPool.erase(tag);
		// a buffer released, notify writer there are available buff now
		m_dstBuffAvailableCond.notify_one();
		LCK2.unlock();

		// record this op in finish set
		std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
		m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
		m_readbyFinishCond.notify_all();
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish readby:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
	}
	else if(myTaskid == m_dstId)
	{
		std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
		std::cv_status w_ret = std::cv_status::no_timeout;
		while(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
		{
			// no tag in buffpool, means the message hasn't come yet
			// wait for writer insert the message to conduit buffpool
			w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
			if(w_ret == std::cv_status::timeout)
			{
				std::cout<<"Error, reader wait time out!"<<"dst-thread "<<myThreadRank<<std::endl;
				LCK2.unlock();
				exit(1);
			}
		}
		// find the tag in the pool, the msg has come, but data may not finish transfer
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
		BuffInfo* tmp_buff = m_srcBuffPool[tag];
		assert(m_srcAvailableBuffCount < m_capacity);
		assert(m_srcBuffPool[tag]->callingReadThreadCount == 0);
		m_srcBuffPool[tag]->callingReadThreadCount = 1;
		// get access lock, will block for waiting writer transferring
		std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buff->buffIdx]);
		// release buff manager lock to allow other thread doing read or write
		LCK2.unlock();
		assert(m_srcBuffWrittenFlag[tmp_buff->buffIdx] == 1);
		// real data transfer
		memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
		LCK3.unlock();

		// need release buff after read
		LCK2.lock();
		while(tmp_buff->safeReleaseAfterRead == false)
		{
			m_srcBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);
		}
		assert(tmp_buff->dataPtr!=nullptr);
		free(tmp_buff->dataPtr);
		tmp_buff->dataPtr = nullptr;
		m_srcBuffWrittenFlag[tmp_buff->buffIdx] =0;
		m_srcBuffIdx.push_back(tmp_buff->buffIdx);
		delete tmp_buff;
		m_srcAvailableBuffCount++;
		m_srcBuffPool.erase(tag);
		// a buffer released, notify writer there are available buff now
		m_srcBuffAvailableCond.notify_one();
		LCK2.unlock();

		// record this op in finish set
		std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
		m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
		m_readbyFinishCond.notify_all();
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish readby:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
	}
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// end ReadBy()

void Conduit::ReadBy_Finish(int tag)
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
		// we use (tag<<LOG_MAX_TASKS)+myTaskid as the internal tag here,
		// as for src and dst we use one finish set, so to need differentiate
		// the srctag and dsttag

		// tag not in finishset, so not finish yet
		m_readbyFinishCond.wait(LCK1);
	}
	// find tag in finishset
	assert(m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
	m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
	if(m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
	{
		//last thread do the finish check, erase this tag from finish set
		// as we erase this tag, so after all threads call this function,
		// another call with same tag value may cause infinite waiting.
		// In one thread, do not call finish check with same tag for two or more
		// times, unless write a new msg with this tag.
		m_readbyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
	}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" finish wait readby!"<<std::endl;
#endif

	return;

}// end ReadBy_Finish()


}// namespace iUtc
