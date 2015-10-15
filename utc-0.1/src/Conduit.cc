#include "Conduit.h"
#include "ConduitManager.h"
#include "TaskManager.h"
#include "Task.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>


namespace iUtc
{

thread_local std::ofstream *Conduit::m_threadOstream=nullptr;
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

    m_srcBuffIdx.clear();
    m_srcBuffAccessMutex.clear();
    m_srcBuffWcallbyAllCond.clear();
    m_srcBuffWrittenFlag.clear();
    m_dstBuffIdx.clear();
    m_dstBuffAccessMutex.clear();
    m_dstBuffWcallbyAllCond.clear();
    m_dstBuffWrittenFlag.clear();

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

int Conduit::Write(void* DataPtr, int DataSize, int tag)
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
	{// srctask calling write()
		// get write hold lock, block other threads who also is calling write()
		std::unique_lock<std::mutex> LCK1(m_srcHoldOtherthreadsWriteMutex);
		// get manager lock to access buffer pool info
		std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		if(m_srcBuffPool.find(tag) != m_srcBuffPool.end())
		{
			// tag exist in conduit, go on checking how many threads has called write
			// for this tag
			int count = m_srcBuffPool[tag]->callingWriteThreadCount;
			assert(count >= 1);
			if(count != m_numSrcLocalThreads)
			{
				// a late coming thread, increase the calling thread count
				count++;
				m_srcBuffPool[tag]->callingWriteThreadCount=count;
				if(count==m_numSrcLocalThreads)
				{
				    // last writer thread, notify reader they can release this buff
				    m_srcBuffPool[tag]->safeReleaseAfterRead = true;
				    m_srcBuffWcallbyAllCond[m_srcBuffPool[tag]->buffIdx].notify_one();
				}
				LCK2.unlock();
				LCK1.unlock();
				return 0;
			}
			else
			{
				// a real exit tag, go on checking if buffpool full
				while(m_srcAvailableBuffCount == 0)
				{
					m_srcBuffAvailableCond.wait(LCK2);
				}
				// get buff, recheck if tag still exist, maybe he was released
				if(m_srcBuffPool.find(tag) != m_srcBuffPool.end())
				{
					// still exist, this would be an tag reuse error
					std::cout<<"Error, tag resued!"<<std::endl;
					LCK2.unlock();
					LCK1.unlock();
					exit(1);
				}
				else
				{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
					// just released, we can use this tag again
					BuffInfo* tmp_buffinfo = new BuffInfo;
					// allocate space
					tmp_buffinfo->dataPtr = malloc(DataSize);
					// get buff id
					tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
					m_srcBuffIdx.pop_back();
					// set count to 1
					tmp_buffinfo->callingWriteThreadCount = 1;
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
					LCK1.unlock();
					// notify reader to read data. (may not need, as writer hold the access lock
					// during transfer, so even reader find the tag in pool, he will block at getting
					// access lock.
					/*m_srcBuffAccessCond[tmp_buffinfo->buffId].nitify_all(); */
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish write!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				}
			}
		}
		else
		{
			// tag not exist, this is the first coming thread for this write() call
			// go on checking if buffpool full

			while(m_srcAvailableBuffCount == 0)
			{
				m_srcBuffAvailableCond.wait(LCK2);
			}
#ifdef USE_DEBUG_LOG
			PRINT_TIME_NOW(*m_threadOstream)
			*m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

			// recover from wait, has buff for using
			BuffInfo* tmp_buffinfo = new BuffInfo;
			// allocate space
			tmp_buffinfo->dataPtr = malloc(DataSize);
			// get buff id
			tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
			m_srcBuffIdx.pop_back();
			// set count to 1
			tmp_buffinfo->callingWriteThreadCount = 1;
			m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_srcAvailableBuffCount--;
			//*m_threadOstream<< m_srcAvailableBuffCount<<" "<<tmp_buffinfo->buffIdx<<std::endl;
			// get access lock for this buffer to write data
			// noticing here we get access lock before release buffmanager lock
			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
			// release buffmanager lock to allow other threads to get buff
			LCK2.unlock();
			// notify reader that one new item inserted to buff pool
			m_srcNewBuffInsertedCond.notify_all();
			// data transfer
			memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
			m_srcBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
			// release access lock and notify reader to read
			LCK3.unlock();
			LCK1.unlock();
			/*m_srcBuffAccessCond[tmp_buffinfo->buffId].nitify_all();*/
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish write!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		}

	} //end (myTaskid == m_dstId)
	else if(myTaskid == m_dstId)
	{// dsttask calling write()
		std::unique_lock<std::mutex> LCK1(m_dstHoldOtherthreadsWriteMutex);
		std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		if(m_dstBuffPool.find(tag) != m_dstBuffPool.end())
		{
			// tag exist in conduit
			int count = m_dstBuffPool[tag]->callingWriteThreadCount;
			assert(count >=1);
			//*m_threadOstream<<count<<"  "<<m_numDstLocalThreads<<std::endl;
			if(count != m_numDstLocalThreads)
			{
				// a late coming thread, increase the calling thread count
				count++;
				m_dstBuffPool[tag]->callingWriteThreadCount=count;
				if(count==m_numSrcLocalThreads)
                {
                    // last writer thread, notify reader they can release this buff
				    m_dstBuffPool[tag]->safeReleaseAfterRead = true;
                    m_dstBuffWcallbyAllCond[m_dstBuffPool[tag]->buffIdx].notify_one();
                }
				LCK1.unlock();
				LCK2.unlock();
				return 0;
			}
			else
			{
				// a real exist tag, go on checking if buffpool full
				/*while(m_dstAvailableBuffCount == 0)
				{
					m_dstBuffAvailableCond.wait(LCK2);
				}*/
				m_dstBuffAvailableCond.wait(LCK2,
						[=](){return m_dstAvailableBuffCount != 0;});
				// get buff, recheck if tag still exist, maybe he was released
				if(m_dstBuffPool.find(tag) != m_dstBuffPool.end())
				{
					// still exist, this would be an error
					std::cout<<"Error, tag resued!"<<std::endl;
					LCK2.unlock();
					LCK1.unlock();
					exit(1);
				}
				else
				{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
					// just released, we can use this tag again
					BuffInfo* tmp_buffinfo = new BuffInfo;
					// allocate space
					tmp_buffinfo->dataPtr = malloc(DataSize);
					// get buff id
					tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
					m_dstBuffIdx.pop_back();
					// set count to 1
					tmp_buffinfo->callingWriteThreadCount = 1;
					m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
					// decrease availabe buff
					m_dstAvailableBuffCount--;
					// get access lock for this buffer to write data
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
					// release buffmanager lock to allow other threads to get buff
					LCK2.unlock();
					// notify reader that one new item inserted to buff pool
					m_dstNewBuffInsertedCond.notify_all();
					// transfer data
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
					// release access lock
					LCK3.unlock();
					LCK1.unlock();
					/*m_dstBuffAccessCond[tmp_buffinfo->buffId].nitify_all();*/
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish write!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				}
			}
		}
		else
		{
			// tag not exist, this is the first coming thread for this write() call
			// go on checking if buffpool full
			/*while(m_dstAvailableBuffCount == 0)
			{
				m_dstBuffAvailableCond.wait(LCK2);
			}*/
			m_dstBuffAvailableCond.wait(LCK2,
					[=](){return m_dstAvailableBuffCount !=0;});
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			// wake from wait, has buff for using
			BuffInfo* tmp_buffinfo = new BuffInfo;
			// allocate space
			tmp_buffinfo->dataPtr = malloc(DataSize);
			// get buff id
			tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
			m_dstBuffIdx.pop_back();
			// set count to 1
			tmp_buffinfo->callingWriteThreadCount = 1;
			m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_dstAvailableBuffCount--;
			// get access lock for this buffer to write data
			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
			// release buffmanager lock to allow other threads to get buff
			LCK2.unlock();
			// notify reader that one new item inserted to buff pool
			m_dstNewBuffInsertedCond.notify_all();
			// data transfer
			memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
			m_dstBuffWrittenFlag[tmp_buffinfo->buffIdx] =1;
			// release access lock
			LCK3.unlock();
			LCK1.unlock();
			/*m_dstBuffAccessCond[tmp_buffinfo->buffId].nitify_all();*/
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish write!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		}
	} //end (myTaskid == m_dstId)
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// Write()

int Conduit::Read(void* DataPtr, int DataSize, int tag)
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
		//*m_threadOstream<<"dst-here"<<myTaskid<<std::endl;
	}
	if(myTaskid == m_srcId)
	{// src task calling read()
		// get hold lock to block other thread, as only one thread do real read, other threads
		// just wait that thread
		std::unique_lock<std::mutex> LCK1(m_srcHoldOtherthreadsReadMutex);
		// get buffpool manager lock to check if the data has come.
		// in src, it will read data from dst buffer, so operate on dst related mutex and cond
		std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		std::cv_status w_ret = std::cv_status::no_timeout;
		while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
		{
			// no tag in buffpool, means the message hasn't come yet
			// wait for writer insert the message to conduit buffpool
			w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
			if(w_ret == std::cv_status::timeout)
			{
				std::cout<<"Error, reader wait time out!"<<std::endl;
				LCK1.unlock();
				LCK2.unlock();
				exit(1);
			}
		}
		// find the tag in the pool, the msg has come, but data may not finish transfer
		BuffInfo* tmp_buff = m_dstBuffPool[tag];
		assert(m_dstAvailableBuffCount < m_capacity);
		int count = tmp_buff->callingReadThreadCount;
		if(count != 0)
		{
			// not first thread that call read(), so data has already read, this is a
			// lately coming thread
			count++;
			m_dstBuffPool[tag]->callingReadThreadCount = count;
			// the last thread will release this buffer
			if(count == m_numSrcLocalThreads)
			{
			    // wait in case that there are  writer threads not finish,
			    // check if is safe to release
			    while(!tmp_buff->safeReleaseAfterRead)
			    {
			        // there are writer threads not finish, we cannot
			        // release buffer now, wait for them
			        m_dstBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2);
			    }
				// Here we do the "real" release
			    assert(tmp_buff->dataPtr != nullptr);
				free(tmp_buff->dataPtr);
				tmp_buff->dataPtr = nullptr;
				m_dstBuffWrittenFlag[tmp_buff->buffIdx] =0;
				m_dstBuffIdx.push_back(tmp_buff->buffIdx);
				delete tmp_buff;
				m_dstAvailableBuffCount++;
				m_dstBuffPool.erase(tag);
				// a buffer released, notify writer there are available buff now
				m_dstBuffAvailableCond.notify_one();
			}
			LCK1.unlock();
			LCK2.unlock();

			return 0;
		}
		else
		{
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			// the first thread doing read()
			count++;
			m_dstBuffPool[tag]->callingReadThreadCount = count;
			// get access lock, will block for waiting writer transferring
			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buff->buffIdx]);
			// release buff manager lock to allow other thread doing read or write
			LCK2.unlock();
			assert(m_dstBuffWrittenFlag[tmp_buff->buffIdx] == 1);
			// real data transfer
			memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
			LCK3.unlock();
			LCK1.unlock();
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		}

	} // end if srctask
	else if(myTaskid == m_dstId)
	{// dst task calling read()

		// get hold lock to block other thread, as only one thread do real read, other threads
		// just wait that thread
		std::unique_lock<std::mutex> LCK1(m_dstHoldOtherthreadsReadMutex);
		// get buffpool manager lock to check if the data has come.
		// in src, it will read data from dst buffer, so operate on dst related mutex and cond
		std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
		bool w_ret;
		/*while(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
		{
			// no tag in buffpool, means the message hasn't come yet
			// wait for writer insert the message to conduit buffpool
			w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
		}*/
		w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT),
				[=](){return (m_srcBuffPool.find(tag)!= m_srcBuffPool.end());});
		if(w_ret == false)
		{
			std::cout<<"Error, reader wait time out!"<<std::endl;
			LCK1.unlock();
			LCK2.unlock();
			exit(1);
		}
		else
		{
			// find the tag in the pool, the msg has come, but data may not finish transfer
			BuffInfo* tmp_buff = m_srcBuffPool[tag];
			assert(m_srcAvailableBuffCount < m_capacity);
			int count = tmp_buff->callingReadThreadCount;
			if(count != 0)
			{
				// not first thread that call read(), so data has already read, this is a
				// lately coming thread
				count++;
				m_srcBuffPool[tag]->callingReadThreadCount = count;
				// the last thread will auto release this buffer
				if(count == m_numDstLocalThreads)
				{
				    // wait for writer threads finish
				    m_srcBuffWcallbyAllCond[tmp_buff->buffIdx].wait(LCK2,
				            [=](){return tmp_buff->safeReleaseAfterRead == true;});
					// Here we do the "real" release
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
				}
				LCK1.unlock();
				LCK2.unlock();

				return 0;
			}
			else
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
				// the first thread doing read()
				count++;
				m_srcBuffPool[tag]->callingReadThreadCount = count; // set to 1
				// get access lock, will block for waiting writer transferring
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buff->buffIdx]);
				// release buff manager lock to allow other thread doing read
				LCK2.unlock();
				assert(m_srcBuffWrittenFlag[tmp_buff->buffIdx] == 1);
				// real data transfer
				memcpy(DataPtr,tmp_buff->dataPtr, DataSize);
				LCK3.unlock();
				LCK1.unlock();
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			}
		}
	} // end if dsttask
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;
}// read()



}// namespace iUtc




