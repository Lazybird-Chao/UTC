#include "Conduit.h"
#include "TaskManager.h"
#include "Task.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>

namespace iUtc{

/*
 * Unbuffered Blocking write operation.
 * Do not user intermediate buffer, pass address.
 */
int Conduit::PWrite(void* DataPtr, int DataSize, int tag)
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
    	// get write op lock and check op counter value to see if need do real write
		std::unique_lock<std::mutex> LCK1(m_srcWriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		int counteridx = m_srcWriteOpRotateCounterIdx[myThreadRank];
		m_srcWriteOpRotateCounter[counteridx]++;
		if(m_srcWriteOpRotateCounter[counteridx] > 1)
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();

				// last thread update some info related with the buff inserted to buffpool
				/*std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);       ///////////////// TODO: may use buffer's access mutex
				assert(m_srcBuffPool.find(tag) != m_srcBuffPool.end());
				assert(m_srcBuffPool[tag]->callingWriteThreadCount == 1);  //only the real write thread has modified this value
				assert(m_srcBuffPool[tag]->safeReleaseAfterRead == false);*/
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[m_srcBuffPool[tag]->buffIdx]);
				m_srcBuffPool[tag]->callingWriteThreadCount = m_numSrcLocalThreads;
				m_srcBuffPool[tag]->safeReleaseAfterRead = true;
				// in case there is reader thread waiting for this to release buff
				m_srcBuffSafeReleaseCond.notify_all();
				LCK3.unlock();

			}
			else
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
			// firsrt coming thread
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

			if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
			{
				// has buff and tag not exist now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// check and wait reader comes
				/*while(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end())
				{
					// no tag in waitlist, so reader hasn't come, need to wait
					m_srcBuffPoolWaitlistCond.wait(LCK2);
				}
				// wakeup when reader comes
				assert(m_srcBuffPoolWaitlist[tag] == 1);
				m_srcBuffPoolWaitlist.erase(tag);*/
				// no need to do this wait, no matter reader comes or not, just pass address
				if(m_srcBuffPoolWaitlist.find(tag) != m_srcBuffPoolWaitlist.end())
				{
					m_srcBuffPoolWaitlist.erase(tag);
				}
				//
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// get buff id
				tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
				m_srcBuffIdx.pop_back();
				// set write thread count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				// insert this buff to buffpool
				m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_srcAvailableBuffCount--;
				// pass the data address to pool, not malloc a buffer
				tmp_buffinfo->dataPtr = DataPtr;
				tmp_buffinfo->isBuffered = false;
				// set writtenflag
				 m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
				//
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_srcNewBuffInsertedCond.notify_all();

				// as pass address, we need wait reader memcpy finish to return
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].wait(LCK3,
						[=](){return m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] == 1;});
				// wakeup when reader finish copy data
				if(m_numSrcLocalThreads == 1)
				{
					// only one local thread, set this flag here, no chance going to late coming thread process
					tmp_buffinfo->safeReleaseAfterRead = true;
					m_srcBuffSafeReleaseCond.notify_all();
				}
				LCK3.unlock();

				// the first thread finish real write, change finishflag
				LCK1.lock();
				assert(m_srcWriteOpRotateFinishFlag[counteridx] == 0);
				// set finish flag
				m_srcWriteOpRotateFinishFlag[counteridx]++;
				if(m_numSrcLocalThreads == 1)
				{   // only one local thread, reset counter and flag at here
					m_srcWriteOpRotateFinishFlag[counteridx] = 0;
					m_srcWriteOpRotateCounter[counteridx] = 0;
				}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				// notify other late coming threads to exit
				m_srcWriteOpFinishCond.notify_all();
				LCK1.unlock();
				return 0;
			}
		}// end first coming thread
    }
    else if(myTaskid == m_dstId)
    {
    	// get write op lock and check op counter value to see if need do real write
		std::unique_lock<std::mutex> LCK1(m_dstWriteOpCheckMutex);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				LCK1.unlock();

				// last thread update some info related with the buff inserted to buffpool
				/*std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
				assert(m_dstBuffPool.find(tag) != m_dstBuffPool.end());
				assert(m_dstBuffPool[tag]->callingWriteThreadCount == 1);
				assert(m_dstBuffPool[tag]->safeReleaseAfterRead == false);*/
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[m_dstBuffPool[tag]->buffIdx]);
				m_dstBuffPool[tag]->callingWriteThreadCount = m_numDstLocalThreads;
				m_dstBuffPool[tag]->safeReleaseAfterRead = true;
				// in case there is reader thread waiting for this to release buff
				m_dstBuffSafeReleaseCond.notify_all();
				LCK3.unlock();
			}
			else
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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

			if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
			{
				// has buff now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// wait for reader
				/*while(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end())
				{
					m_dstBuffPoolWaitlistCond.wait(LCK2);
				}
				// wakeup when reader comes
				assert(m_dstBuffPoolWaitlist[tag]==1);
				m_dstBuffPoolWaitlist.erase(tag);*/
				if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
				{
					m_dstBuffPoolWaitlist.erase(tag);
				}
				//
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// get buff id
				tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
				m_dstBuffIdx.pop_back();
				// set count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				// insert this buff to buffpool
				m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_dstAvailableBuffCount--;
				// reader is waiting, use address to pass msg
				tmp_buffinfo->dataPtr = DataPtr;
				tmp_buffinfo->isBuffered = false;
				// set written flag here
				m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] = 1;
				// release buffmanager lock to allow other threads to get buff
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_dstNewBuffInsertedCond.notify_all();

				// wait reader finish data copy
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].wait(LCK3,
						[=](){return m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] == 1;});
				if(m_numDstLocalThreads == 1)
				{   // only one local thread, set this flag here, no chance going to late coming thread process
					tmp_buffinfo->safeReleaseAfterRead = true;
					m_dstBuffSafeReleaseCond.notify_all();
				}
				LCK3.unlock();

				// the first thread finish real write
				LCK1.lock();
				assert(m_dstWriteOpRotateFinishFlag[counteridx] == 0);
				// set finish flag
				m_dstWriteOpRotateFinishFlag[counteridx]++;
				if(m_numSrcLocalThreads == 1)
				{   // only one local thread, reset counter and flag at here
					m_dstWriteOpRotateFinishFlag[counteridx] = 0;
					m_dstWriteOpRotateCounter[counteridx] = 0;
				}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// update counter idx to next one
				m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
				// notify other late coming threads to exit
				m_dstWriteOpFinishCond.notify_all();
				LCK1.unlock();
				return 0;
			}
		}// end first coming thread
    }// end dsttask
    else
    {
    	std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    	exit(1);
    }

    return 0;

}// end pwrite()


int Conduit::PWriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
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
        *m_threadOstream<<"thread "<<myThreadRank<<" call Pwriteby..."<<std::endl;
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
        *m_threadOstream<<"thread "<<myThreadRank<<" exit Pwriteby!"<<std::endl;
#endif
        return 0;
    }

    	// current thread is the writing thread
    	if(myTaskid == m_srcId)
    	{
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
    		if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
    		{
    			// has buff and tag not exist, do real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwriteby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				/*while(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end())
				{
					m_srcBuffPoolWaitlistCond.wait(LCK2);
				}
				assert(m_srcBuffPoolWaitlist[tag]==1);
				m_srcBuffPoolWaitlist.erase(tag);*/
				if(m_srcBuffPoolWaitlist.find(tag) != m_srcBuffPoolWaitlist.end())
				{
					m_srcBuffPoolWaitlist.erase(tag);
				}
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// get buff id
				tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
				m_srcBuffIdx.pop_back();
				// set count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				// insert this buff to buffpool
				m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_srcAvailableBuffCount--;
				// pass address
				tmp_buffinfo->dataPtr = DataPtr;
				tmp_buffinfo->isBuffered = false;
				m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] = 1;
				LCK2.unlock();
				m_srcNewBuffInsertedCond.notify_all();

				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].wait(LCK3,
						[=](){return m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] == 1;});
				// wakeup when reader finish copy data
				// only one thread do write, set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				m_srcBuffSafeReleaseCond.notify_all();
				LCK3.unlock();

				// record this op to readby finish set
				std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
				//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
				//      m_numSrcLocalThreads-1));
				m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
				m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwriteby!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				return 0;
    		}
    	}// end srctask
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
			if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
			{
				// has buff now, can do the real write
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwriteby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				/*while(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end())
				{
					m_dstBuffPoolWaitlistCond.wait(LCK2);
				}
				assert(m_dstBuffPoolWaitlist[tag] ==1);
				m_dstBuffPoolWaitlist.erase(tag);*/
				if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
				{
					m_dstBuffPoolWaitlist.erase(tag);
				}
				BuffInfo* tmp_buffinfo = new BuffInfo;
				// get buff id
				tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
				m_dstBuffIdx.pop_back();
				// set count to 1
				tmp_buffinfo->callingWriteThreadCount = 1;
				// insert this buff to buffpool
				m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
				// decrease availabe buff
				m_dstAvailableBuffCount--;
				//
				tmp_buffinfo->dataPtr = DataPtr;
				tmp_buffinfo->isBuffered = false;
				m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] = 1;
				LCK2.unlock();
				m_dstNewBuffInsertedCond.notify_all();

				// wait reader finish data copy
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].wait(LCK3,
						[=](){return m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] == 1;});
				// set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				m_dstBuffSafeReleaseCond.notify_all();
				LCK3.unlock();

				// record this op to readby finish set
				std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
				//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
				//      m_numDstLocalThreads-1));
				m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
				m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwriteby!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				return 0;
			}
    	}// end dsttask
    	else
    	{
    		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    		exit(1);
    	}

}// end pwriteby()


void Conduit::PWriteBy_Finish(int tag)
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
		m_writebyFinishCond.wait(LCK1);
	}
	// find tag in finishset
	assert(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
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
}// end pwriteby_finish()


}// end namespace iUtc
