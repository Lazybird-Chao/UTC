#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
#include "Task_Utilities.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>


namespace iUtc
{


/*
 *  Buffered & Blocking conduit write operation
 */
int InprocConduit::BWrite(void *DataPtr, int DataSize, int tag)
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        int counteridx = m_srcWriteOpRotateCounterIdx[myThreadRank];
        m_srcWriteOpRotateCounter[counteridx]++;
        if(m_srcWriteOpRotateCounter[counteridx] >1)
        {
            // a late coming thread, but at most = all local threads
#ifdef USE_DEBUG_ASSERT
            assert(m_srcWriteOpRotateCounter[counteridx] <= m_numSrcLocalThreads);
#endif

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
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            }
            else
            {
                // update counter idx to next one
                m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
                LCK1.unlock();
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            }
            return 0;

        }
        else
        {
            LCK1.unlock();
            // the first coming thread, who will do real write
            BuffInfo* tmp_buffinfo;
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
                std::cerr<<"Error, tag resued!"<<std::endl;
                LCK2.unlock();
                exit(1);
            }

            if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
            {
                // has buff and tag not exist now, can do the real write

                if(m_srcBuffPoolWaitlist.find(tag)!= m_srcBuffPoolWaitlist.end())
                {
                    // tag is in waitlist, means reader is already waiting for this msg.
                    // for buffered write, this doesn't matter
                    m_srcBuffPoolWaitlist.erase(tag);
                }
                // create buffer and insert to pool
                BuffInfo* tmp_buffinfo = new BuffInfo;
                // allocate space
                tmp_buffinfo->dataPtr = malloc(DataSize);
                if(!tmp_buffinfo->dataPtr)
                	std::cerr<<"Error, not enough memory!"<<std::endl;
                // get buff id
                tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
                m_srcBuffIdx.pop_back();
                // set write thread count to 1
                tmp_buffinfo->callingWriteThreadCount = 1;
                // insert this buff to buffpool
                m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                // decrease availabe buff
                m_srcAvailableBuffCount--;
                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
					if(m_numSrcLocalThreads == 1)
					{
						// only one local thread, set this flag here, no chance going to late coming thread process
						tmp_buffinfo->safeReleaseAfterRead = true;
						// no need to call wcallbyallcond.notify, as no reader thread's release can happen before this time point
					}
					LCK2.unlock();
					m_srcNewBuffInsertedCond.notify_all();
                }
                else
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
					// release buffmanager lock to allow other threads to get buff
					LCK2.unlock();
					// notify reader that one new item inserted to buff pool
					m_srcNewBuffInsertedCond.notify_all();

					// get access lock for this buffer to write data
					std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
					// do real data transfer
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
					if(m_numSrcLocalThreads == 1)
					{
						// only one local thread, set this flag here, no chance going to late coming thread process
						tmp_buffinfo->safeReleaseAfterRead = true;
						// no need to call wcallbyallcond.notify, as no reader thread's release can happen before this time point
					}
					// release access lock
					LCK3.unlock();
					// notify reader to read data
					m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].notify_one();
                }

                // the first thread finish real write, change finishflag
                LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                assert(m_srcWriteOpRotateFinishFlag[counteridx] == 0);
#endif
                // set finish flag
                m_srcWriteOpRotateFinishFlag[counteridx]++;
                if(m_numSrcLocalThreads == 1)
                {   // only one local thread, reset counter and flag at here
                    m_srcWriteOpRotateFinishFlag[counteridx] = 0;
                    m_srcWriteOpRotateCounter[counteridx] = 0;
                }
                // update counter idx to next one
                m_srcWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
                // notify other late coming threads to exit
                m_srcWriteOpFinishCond.notify_all();
                LCK1.unlock();
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Bwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        int counteridx = m_dstWriteOpRotateCounterIdx[myThreadRank];
        m_dstWriteOpRotateCounter[counteridx]++;
        if(m_dstWriteOpRotateCounter[counteridx] >1)
        {
            // a late coming thread, but at most = all local threads
#ifdef USE_DEBUG_ASSERT
            assert(m_dstWriteOpRotateCounter[counteridx] <= m_numDstLocalThreads);
#endif

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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                // update counter idx to next one
                m_dstWriteOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_capacity+1);
                LCK1.unlock();

                // last thread update some info related with the buff inserted to buffpool
                /*std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
                ////////// TODO: may use access mutex, then no need to race this manager mutex, but reader side who will check safe release flag
                ////////// and do release may has problem, as he need managermutex to release a buffer
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
                std::cerr<<"Error, tag resued!"<<std::endl;
                LCK2.unlock();
                exit(1);
            }

            if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
            {
                // has buff now, can do the real write

                if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
                {
                    m_dstBuffPoolWaitlist.erase(tag);
                }

                BuffInfo* tmp_buffinfo = new BuffInfo;
                // allocate space
                tmp_buffinfo->dataPtr = malloc(DataSize);
                if(!tmp_buffinfo->dataPtr)
					std::cerr<<"Error, not enough memory!"<<std::endl;
                // get buff id
                tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
                m_dstBuffIdx.pop_back();
                // set count to 1
                tmp_buffinfo->callingWriteThreadCount = 1;
                // insert this buff to buffpool
                m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                // decrease availabe buff
                m_dstAvailableBuffCount--;
                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
					if(m_numDstLocalThreads == 1)
					{   // only one local thread, set this flag here, no chance going to late coming thread process
						tmp_buffinfo->safeReleaseAfterRead = true;
					}
					LCK2.unlock();
					// notify reader that one new item inserted to buff pool
					m_dstNewBuffInsertedCond.notify_all();
                }
                else
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
					// release buffmanager lock to allow other threads to get buff
					LCK2.unlock();
					// notify reader that one new item inserted to buff pool
					m_dstNewBuffInsertedCond.notify_all();

					// get access lock for this buffer to write data
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
					// do real data transfer
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
					if(m_numDstLocalThreads == 1)
					{   // only one local thread, set this flag here, no chance going to late coming thread process
						tmp_buffinfo->safeReleaseAfterRead = true;
					}
					// release access lock
					LCK3.unlock();
					// notify reader to read data.
					m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].notify_one();
                }

                // the first thread finish real write
                LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                assert(m_dstWriteOpRotateFinishFlag[counteridx] == 0);
#endif
                // set finish flag
                m_dstWriteOpRotateFinishFlag[counteridx]++;
                if(m_numSrcLocalThreads == 1)
                {   // only one local thread, reset counter and flag at here
                    m_dstWriteOpRotateFinishFlag[counteridx] = 0;
                    m_dstWriteOpRotateCounter[counteridx] = 0;
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Bwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }
}// end Write()

int InprocConduit::BWriteBy(ThreadRank thread, void *DataPtr, int DataSize, int tag)
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
        *m_threadOstream<<"thread "<<myThreadRank<<" call Bwriteby..."<<std::endl;
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
        *m_threadOstream<<"thread "<<myThreadRank<<" exit Bwriteby!"<<std::endl;
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
            std::cerr<<"Error, tag resued!"<<std::endl;
            LCK2.unlock();
            exit(1);
        }

        if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
        {
            // has buff and tag not exist, do real write

            if(m_srcBuffPoolWaitlist.find(tag) != m_srcBuffPoolWaitlist.end())
            {
                m_srcBuffPoolWaitlist.erase(tag);
            }
            BuffInfo* tmp_buffinfo = new BuffInfo;
            // allocate space
            tmp_buffinfo->dataPtr = malloc(DataSize);
            if(!tmp_buffinfo->dataPtr)
				std::cerr<<"Error, not enough memory!"<<std::endl;
            // get buff id
            tmp_buffinfo->buffIdx = m_srcBuffIdx.back();
            m_srcBuffIdx.pop_back();
            // set count to 1
            tmp_buffinfo->callingWriteThreadCount = 1;
            // insert this buff to buffpool
            m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
            // decrease availabe buff
            m_srcAvailableBuffCount--;
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwriteby small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// only one thread do write, set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_srcNewBuffInsertedCond.notify_all();
            }
            else
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwriteby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				// release buffmanager lock to allow other threads to get buff
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_srcNewBuffInsertedCond.notify_all();

				// get access lock for this buffer to write data
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				// do real data transfer
				memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// only one thread do write, set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				// release access lock
				LCK3.unlock();
				m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].notify_one();
            }

            // record this op to readby finish set
            std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
            //m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
            //      m_numSrcLocalThreads-1));
            m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
            m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Bwriteby!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
            std::cerr<<"Error, tag resued!"<<std::endl;
            LCK2.unlock();
            exit(1);
        }

        if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
        {
            // has buff now, can do the real write

            if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
            {
                // tag is in the waitlist, reader already waits there
                m_dstBuffPoolWaitlist.erase(tag);
            }
            BuffInfo* tmp_buffinfo = new BuffInfo;
            // allocate space
            tmp_buffinfo->dataPtr = malloc(DataSize);
            if(!tmp_buffinfo->dataPtr)
            	std::cerr<<"Error, not enough memory!"<<std::endl;
            // get buff id
            tmp_buffinfo->buffIdx = m_dstBuffIdx.back();
            m_dstBuffIdx.pop_back();
            // set count to 1
            tmp_buffinfo->callingWriteThreadCount = 1;
            // insert this buff to buffpool
            m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
            // decrease availabe buff
            m_dstAvailableBuffCount--;
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwriteby small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_dstNewBuffInsertedCond.notify_all();
            }
            else
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwriteby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
				// release buffmanager lock to allow other threads to get buff
				LCK2.unlock();
				// notify reader that one new item inserted to buff pool
				m_dstNewBuffInsertedCond.notify_all();

				// get access lock for this buffer to write data
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				// do real data transfer
				memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
				// set safe release flag
				tmp_buffinfo->safeReleaseAfterRead = true;
				// release access lock
				LCK3.unlock();
				m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].notify_one();
            }

            // record this op to readby finish set
            std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
            //m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
            //      m_numDstLocalThreads-1));
            m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
            m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Bwriteby!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            return 0;
        }
    }
    else
    {
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;

}// end BWriteBy()

void InprocConduit::BWriteBy_Finish(int tag)
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
        *m_threadOstream<<"thread "<<myThreadRank<<" wait for Bwriteby..."<<std::endl;
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
#ifdef USE_DEBUG_ASSERT
    assert(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]>0);
#endif
    m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]--;
    if(m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid]==0)
    {
        //last thread do the finish check, erase this tag from finish set
        // as we erase this tag, so after all threads call this function,
        // another call with same tag value may cause infinite waiting.
        // In a thread, do not call finish check with same tag for two or more
        // times, unless write a new msg with this tag.
        m_writebyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
    }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" finish wait Bwriteby!"<<std::endl;
#endif

    return;

}// end BWriteBy_Finish




}// end namespace iUtc
