#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
#include "Task_Utilities.h"
#include "UtcBasics.h"


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
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        if(m_numSrcLocalThreads ==1)
        {
            // only one local thread for the task
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
                // get buff id
                int tmp_idx = m_srcBuffIdx.back();
                m_srcBuffIdx.pop_back();
                BuffInfo* tmp_buffinfo = &(m_srcAvailableBuff[tmp_idx]);
                tmp_buffinfo->buffIdx = tmp_idx;
                // set write thread count to 1
                tmp_buffinfo->callingWriteThreadCount = 1;
                // insert this buff to buffpool
                m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                // decrease availabe buff
                m_srcAvailableBuffCount--;
                // alloc memory buffer for storing data
                int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
                tmp_size *= CONDUIT_BUFFER_SIZE;
                if(tmp_buffinfo->buffSize == tmp_size)
                {
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else if(tmp_buffinfo->buffSize < tmp_size)
                {
                    free(tmp_buffinfo->bufferPtr);
                    tmp_buffinfo->buffSize = tmp_size;
                    tmp_buffinfo->bufferPtr = malloc(tmp_size);
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else
                {
                    if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                    {
                        tmp_buffinfo->buffSize = tmp_size;
                        tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                        tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else
                    {
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor++;
                    }
                }
                // copy data to buffer
                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                    memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                    m_srcBuffDataWrittenFlag[tmp_idx] =1;
                    LCK2.unlock();
                    m_srcNewBuffInsertedCond.notify_all();
                }
                else
                {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite big msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                    // release buffmanager lock to allow other threads to get buff
                    LCK2.unlock();
                    // notify reader that one new item inserted to buff pool
                    m_srcNewBuffInsertedCond.notify_all();

                    // get access lock for this buffer to write data
                    std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
                    // do real data transfer
                    memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                    m_srcBuffDataWrittenFlag[tmp_idx] =1;
                    // release access lock
                    LCK3.unlock();
                    // notify reader to read data
                    m_srcBuffDataWrittenCond[tmp_idx].notify_one();
                }// end big msg
            }

        }// end for one thread
        else
        {
            // there are several local threads

            // get op check lock
            std::unique_lock<std::mutex> LCK1(m_srcOpCheckMutex);
            int counteridx = m_srcOpRotateCounterIdx[myThreadRank];
            m_srcOpRotateCounter[counteridx]++;
            if(m_srcOpRotateCounter[counteridx] >1)
            {
                // a late coming thread, but at most = all local threads
#ifdef USE_DEBUG_ASSERT
                assert(m_srcOpRotateCounter[counteridx] <= m_numSrcLocalThreads);
#endif

                while(m_srcOpRotateFinishFlag[counteridx] == 0)
                {
                    // the first thread which do real write hasn't finish, we will wait for it to finish
                    m_srcOpFinishCond.wait(LCK1);
                }
                // wake up, so the write is finished
                m_srcOpRotateFinishFlag[counteridx]++;
                if(m_srcOpRotateFinishFlag[counteridx] == m_numSrcLocalThreads)
                {
                    // last thread that will leave this write, reset counter and flag value
                    m_srcOpRotateFinishFlag[counteridx] = 0;
                    m_srcOpRotateCounter[counteridx] = 0;
                    // update counter idx to next one
                    m_srcOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);

                    m_srcAvailableNoFinishedOpCount++;
                    m_srcAvailableNoFinishedOpCond.notify_one();
                    LCK1.unlock();
                }
                else
                {
                    // update counter idx to next one
                    m_srcOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
                    LCK1.unlock();
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

                return 0;
            }// end for late coming thread
            else
            {

                // first coming thread
                while(m_srcAvailableNoFinishedOpCount ==0)
                {
                    m_srcAvailableNoFinishedOpCond.wait(LCK1);
                }
                m_srcAvailableNoFinishedOpCount--;
                LCK1.unlock();

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
                    // get buff id
                    int tmp_idx = m_srcBuffIdx.back();
                    m_srcBuffIdx.pop_back();
                    BuffInfo *tmp_buffinfo = &(m_srcAvailableBuff[tmp_idx]);
                    tmp_buffinfo->buffIdx = tmp_idx;
                    // set write thread count to 1
                    tmp_buffinfo->callingWriteThreadCount = 1;
                    // insert this buff to buffpool
                    m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                    // decrease availabe buff
                    m_srcAvailableBuffCount--;
                    // alloc memory
                    int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
                    tmp_size *= CONDUIT_BUFFER_SIZE;
                    if(tmp_buffinfo->buffSize == tmp_size)
                    {
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else if(tmp_buffinfo->buffSize < tmp_size)
                    {
                        free(tmp_buffinfo->bufferPtr);
                        tmp_buffinfo->buffSize = tmp_size;
                        tmp_buffinfo->bufferPtr = malloc(tmp_size);
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else
                    {
                        if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                        {
                            tmp_buffinfo->buffSize = tmp_size;
                            tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                            tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                            tmp_buffinfo->reduceBuffsizeSensor = 0;
                        }
                        else
                        {
                            tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                            tmp_buffinfo->reduceBuffsizeSensor++;
                        }
                    }

                    if(DataSize<SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                        memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                        m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
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
                        std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
                        // do real data transfer
                        memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                        m_srcBuffDataWrittenFlag[tmp_idx] =1;
                        // release access lock
                        LCK3.unlock();
                        // notify reader to read data
                        m_srcBuffDataWrittenCond[tmp_idx].notify_one();
                    }

                    // the first thread finish real write, change finishflag
                    LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                    assert(m_srcOpRotateFinishFlag[counteridx] == 0);
#endif
                    // set finish flag
                    m_srcOpRotateFinishFlag[counteridx]++;
                    // update counter idx to next one
                    m_srcOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
                    // notify other late coming threads to exit
                    m_srcOpFinishCond.notify_all();
                    LCK1.unlock();
                }
            }// end first thread

        }// end for sevral threads

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Bwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
         return 0;
    }//end srcTask
    else if(myTaskid == m_dstId)
    {
        // dsttask calling write()
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads ==1)
        {
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
                // get buff id
                int tmp_idx = m_dstBuffIdx.back();
                m_dstBuffIdx.pop_back();
                BuffInfo *tmp_buffinfo = &(m_dstAvailableBuff[tmp_idx]);
                tmp_buffinfo->buffIdx = tmp_idx;
                // set count to 1
                tmp_buffinfo->callingWriteThreadCount = 1;
                // insert this buff to buffpool
                m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                // decrease availabe buff
                m_dstAvailableBuffCount--;
                // alloc memory
                int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
                tmp_size *= CONDUIT_BUFFER_SIZE;
                if(tmp_buffinfo->buffSize == tmp_size)
                {
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else if(tmp_buffinfo->buffSize < tmp_size)
                {
                    free(tmp_buffinfo->bufferPtr);
                    tmp_buffinfo->buffSize = tmp_size;
                    tmp_buffinfo->bufferPtr = malloc(tmp_size);
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else
                {
                    if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                    {
                        tmp_buffinfo->buffSize = tmp_size;
                        tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                        tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else
                    {
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor++;
                    }
                }

                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                    memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                    m_dstBuffDataWrittenFlag[tmp_idx] =1;
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
                    std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
                    // do real data transfer
                    memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                    m_dstBuffDataWrittenFlag[tmp_idx] =1;
                    // release access lock
                    LCK3.unlock();
                    // notify reader to read data.
                    m_dstBuffDataWrittenCond[tmp_idx].notify_one();
                }// end big msg
            }

        }// end for one thread
        else
        {
            // get write op lock and check op counter value to see if need do real write
            std::unique_lock<std::mutex> LCK1(m_dstOpCheckMutex);

            int counteridx = m_dstOpRotateCounterIdx[myThreadRank];
            m_dstOpRotateCounter[counteridx]++;
            if(m_dstOpRotateCounter[counteridx] >1)
            {
                // a late coming thread, but at most = all local threads
#ifdef USE_DEBUG_ASSERT
                assert(m_dstOpRotateCounter[counteridx] <= m_numDstLocalThreads);
#endif

                while(m_dstOpRotateFinishFlag[counteridx] == 0)
                {
                    // the first thread which do real write hasn't finish, we will wait for it to finish
                    m_dstOpFinishCond.wait(LCK1);
                }
                // wake up, so the write is finished
                m_dstOpRotateFinishFlag[counteridx]++;
                if(m_dstOpRotateFinishFlag[counteridx] == m_numDstLocalThreads)
                {
                    // last thread that will leave this write, reset counter and flag value
                    m_dstOpRotateFinishFlag[counteridx] = 0;
                    m_dstOpRotateCounter[counteridx] = 0;
                    // update counter idx to next one
                    m_dstOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);

                    m_dstAvailableNoFinishedOpCount++;
                    m_dstAvailableNoFinishedOpCond.notify_one();
                    LCK1.unlock();

                }
                else
                {
                    // update counter idx to next one
                    m_dstOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
                    LCK1.unlock();
                }

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                return 0;
            }// end late threads
            else
            {
                // the first coming thread, who will do real write
                while(m_dstAvailableNoFinishedOpCount == 0)
                {
                    m_dstAvailableNoFinishedOpCond.wait(LCK1);
                }
                m_dstAvailableNoFinishedOpCount--;
                LCK1.unlock();

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
                    // get buff id
                    int tmp_idx = m_dstBuffIdx.back();
                    m_dstBuffIdx.pop_back();
                    BuffInfo* tmp_buffinfo=&(m_dstAvailableBuff[tmp_idx]);
                    // set count to 1
                    tmp_buffinfo->callingWriteThreadCount = 1;
                    // insert this buff to buffpool
                    m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
                    // decrease availabe buff
                    m_dstAvailableBuffCount--;
                    int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
                    tmp_size *= CONDUIT_BUFFER_SIZE;
                    if(tmp_buffinfo->buffSize == tmp_size)
                    {
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else if(tmp_buffinfo->buffSize < tmp_size)
                    {
                        free(tmp_buffinfo->bufferPtr);
                        tmp_buffinfo->buffSize = tmp_size;
                        tmp_buffinfo->bufferPtr = malloc(tmp_size);
                        tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                        tmp_buffinfo->reduceBuffsizeSensor = 0;
                    }
                    else
                    {
                        if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                        {
                            tmp_buffinfo->buffSize = tmp_size;
                            tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                            tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                            tmp_buffinfo->reduceBuffsizeSensor = 0;
                        }
                        else
                        {
                            tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                            tmp_buffinfo->reduceBuffsizeSensor++;
                        }
                    }
                    if(DataSize<SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                        memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                        m_dstBuffDataWrittenFlag[tmp_idx] =1;
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
                        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
                        // do real data transfer
                        memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
                        m_dstBuffDataWrittenFlag[tmp_idx] =1;
                        // release access lock
                        LCK3.unlock();
                        // notify reader to read data.
                        m_dstBuffDataWrittenCond[tmp_idx].notify_one();
                    }

                    // the first thread finish real write
                    LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                    assert(m_dstOpRotateFinishFlag[counteridx] == 0);
#endif
                    // set finish flag
                    m_dstOpRotateFinishFlag[counteridx]++;
                    // update counter idx to next one
                    m_dstOpRotateCounterIdx[myThreadRank] = (counteridx+1)%(m_noFinishedOpCapacity+1);
                    // notify other late coming threads to exit
                    m_dstOpFinishCond.notify_all();
                    LCK1.unlock();
                }
            }// end first thread threads
        }// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Bwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        return 0;

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
            // get buff id
            int tmp_idx = m_srcBuffIdx.back();
            m_srcBuffIdx.pop_back();
            BuffInfo *tmp_buffinfo = &(m_srcAvailableBuff[tmp_idx]);
            tmp_buffinfo->buffIdx = tmp_idx;
            // set count to 1
            tmp_buffinfo->callingWriteThreadCount = 1;
            // insert this buff to buffpool
            m_srcBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
            // decrease availabe buff
            m_srcAvailableBuffCount--;
            int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
            tmp_size *= CONDUIT_BUFFER_SIZE;
            if(tmp_buffinfo->buffSize == tmp_size)
            {
                tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                tmp_buffinfo->reduceBuffsizeSensor = 0;
            }
            else if(tmp_buffinfo->buffSize < tmp_size)
            {
                free(tmp_buffinfo->bufferPtr);
                tmp_buffinfo->buffSize = tmp_size;
                tmp_buffinfo->bufferPtr = malloc(tmp_size);
                tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                tmp_buffinfo->reduceBuffsizeSensor = 0;
            }
            else
            {
                if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                {
                    tmp_buffinfo->buffSize = tmp_size;
                    tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                    tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else
                {
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor++;
                }
            }
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwriteby small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_srcBuffDataWrittenFlag[tmp_idx] =1;
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
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
				// do real data transfer
				memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_srcBuffDataWrittenFlag[tmp_idx] =1;
				// release access lock
				LCK3.unlock();
				m_srcBuffDataWrittenCond[tmp_idx].notify_one();
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
            // get buff id
            int tmp_idx = m_dstBuffIdx.back();
            m_dstBuffIdx.pop_back();
            BuffInfo* tmp_buffinfo = &(m_dstAvailableBuff[tmp_idx]);
            tmp_buffinfo->buffIdx = tmp_idx;
            // set count to 1
            tmp_buffinfo->callingWriteThreadCount = 1;
            // insert this buff to buffpool
            m_dstBuffPool.insert(std::pair<MessageTag, BuffInfo*>(tag, tmp_buffinfo));
            // decrease availabe buff
            m_dstAvailableBuffCount--;

            int tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
            tmp_size *= CONDUIT_BUFFER_SIZE;
            if(tmp_buffinfo->buffSize == tmp_size)
            {
                tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                tmp_buffinfo->reduceBuffsizeSensor = 0;
            }
            else if(tmp_buffinfo->buffSize < tmp_size)
            {
                free(tmp_buffinfo->bufferPtr);
                tmp_buffinfo->buffSize = tmp_size;
                tmp_buffinfo->bufferPtr = malloc(tmp_size);
                tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                tmp_buffinfo->reduceBuffsizeSensor = 0;
            }
            else
            {
                if(tmp_buffinfo->reduceBuffsizeSensor > 3)
                {
                    tmp_buffinfo->buffSize = tmp_size;
                    tmp_buffinfo->dataPtr = realloc(tmp_buffinfo->bufferPtr, tmp_size);
                    tmp_buffinfo->bufferPtr = tmp_buffinfo->dataPtr;
                    tmp_buffinfo->reduceBuffsizeSensor = 0;
                }
                else
                {
                    tmp_buffinfo->dataPtr = tmp_buffinfo->bufferPtr;
                    tmp_buffinfo->reduceBuffsizeSensor++;
                }
            }

            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwriteby small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            	memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
				m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =1;
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
