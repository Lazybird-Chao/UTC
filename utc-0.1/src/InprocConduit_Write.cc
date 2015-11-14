#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
#include "Task_Utilities.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>


namespace iUtc{


/*
 * Standard Blocking write operation.
 * Do buffered message when it needs to.
 */
int InprocConduit::Write(void* DataPtr, int DataSize, int tag)
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
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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

                if(m_srcBuffPoolWaitlist.find(tag)!= m_srcBuffPoolWaitlist.end())
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write ptr...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                    // tag is in waitlist, means reader is already waiting for this msg.
                    // passing address
                    m_srcBuffPoolWaitlist.erase(tag);

                    tmp_buffinfo->dataPtr = DataPtr;
                    m_srcBuffDataWrittenFlag[tmp_idx] = 1;
                    LCK2.unlock();
                    m_srcNewBuffInsertedCond.notify_all();

                    // wait reader finish data copy
                    std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
                    m_srcBuffDataReadCond[tmp_idx].wait(LCK3,
                            [=](){return m_srcBuffDataReadFlag[tmp_idx] == 1;});
                    LCK3.unlock();

                    LCK2.lock();
                    m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                    tmp_buffinfo->dataPtr= nullptr;
                    tmp_buffinfo->callingReadThreadCount=0;
                    tmp_buffinfo->callingWriteThreadCount=0;
                    tmp_buffinfo->reduceBuffsizeSensor=0;
                    m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
                    tmp_buffinfo->buffIdx = -1;
                    m_srcAvailableBuffCount++;
                    m_srcBuffPool.erase(tag);
                    m_srcBuffAvailableCond.notify_one();
                    LCK2.unlock();

                }
                else
                {
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
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write big msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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

                }// end for using intermediate buffer
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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

                    if(m_srcBuffPoolWaitlist.find(tag)!= m_srcBuffPoolWaitlist.end())
                    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write ptr...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                        // tag is in waitlist, means reader is already waiting for this msg.
                        // for buffered write, this doesn't matter
                        m_srcBuffPoolWaitlist.erase(tag);
                        tmp_buffinfo->dataPtr = DataPtr;
                        m_srcBuffDataWrittenFlag[tmp_idx]=1;
                        LCK2.unlock();
                        m_srcNewBuffInsertedCond.notify_all();

                        // as pass address, we need wait reader memcpy finish to return
                        std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
                        m_srcBuffDataReadCond[tmp_idx].wait(LCK3,
                                [=](){return m_srcBuffDataReadFlag[tmp_idx] == 1;});
                        // wakeup when reader finish copy data
                        LCK3.unlock();

                        LCK2.lock();
                        m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                        m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                        tmp_buffinfo->dataPtr= nullptr;
                        tmp_buffinfo->callingReadThreadCount=0;
                        tmp_buffinfo->callingWriteThreadCount=0;
                        tmp_buffinfo->reduceBuffsizeSensor=0;
                        m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
                        tmp_buffinfo->buffIdx = -1;
                        m_srcAvailableBuffCount++;
                        m_srcBuffPool.erase(tag);
                        m_srcBuffAvailableCond.notify_one();
                        LCK2.unlock();
                    }
                    else
                    {
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing write...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish write!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
         return 0;

    }// end srctask
    else if(myTaskid == m_dstId)
    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads ==1)
        {
            // check if there is available buff
            std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
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

                if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
                {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write by ptr...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                    m_dstBuffPoolWaitlist.erase(tag);
                    // use address to pass msg
                    tmp_buffinfo->dataPtr = DataPtr;
                    // set written flag here
                    m_dstBuffDataWrittenFlag[tmp_idx] = 1;
                    // release buffmanager lock to allow other threads to get buff
                    LCK2.unlock();
                    // notify reader that one new item inserted to buff pool
                    m_dstNewBuffInsertedCond.notify_all();

                    // wait reader finish data copy
                    std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
                    m_dstBuffDataReadCond[tmp_idx].wait(LCK3,
                            [=](){return m_dstBuffDataReadFlag[tmp_idx] == 1;});
                    LCK3.unlock();

                    // release buff
                    LCK2.lock();
                    m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                    tmp_buffinfo->dataPtr= nullptr;
                    tmp_buffinfo->callingReadThreadCount=0;
                    tmp_buffinfo->callingWriteThreadCount=0;
                    tmp_buffinfo->reduceBuffsizeSensor=0;
                    m_dstBuffIdx.push_back(tmp_buffinfo->buffIdx);
                    tmp_buffinfo->buffIdx = -1;
                    m_dstAvailableBuffCount++;
                    m_dstBuffPool.erase(tag);
                    m_dstBuffAvailableCond.notify_one();
                    LCK2.unlock();
                }
                else
                {
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
                } //end of using intermediate buffer
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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

                    if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
                    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write ptr...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                        m_dstBuffPoolWaitlist.erase(tag);
                        tmp_buffinfo->dataPtr = DataPtr;
                        m_dstBuffDataWrittenFlag[tmp_idx]=1;
                        LCK2.unlock();
                        m_dstNewBuffInsertedCond.notify_all();

                        // wait reader finish data copy
                        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
                        m_dstBuffDataReadCond[tmp_idx].wait(LCK3,
                                [=](){return m_dstBuffDataReadFlag[tmp_idx] == 1;});
                        LCK3.unlock();

                        // release buff
                        LCK2.lock();
                        m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                        m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                        tmp_buffinfo->dataPtr= nullptr;
                        tmp_buffinfo->callingReadThreadCount=0;
                        tmp_buffinfo->callingWriteThreadCount=0;
                        tmp_buffinfo->reduceBuffsizeSensor=0;
                        m_dstBuffIdx.push_back(tmp_buffinfo->buffIdx);
                        tmp_buffinfo->buffIdx = -1;
                        m_dstAvailableBuffCount++;
                        m_dstBuffPool.erase(tag);
                        m_dstBuffAvailableCond.notify_one();
                        LCK2.unlock();

                    }
                    else
                    {
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing write...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
                    }// end of using intermediate buffer

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

    }// end dsttask
    else
    {
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    	exit(1);
    }

    return 0;
}// end Write()


int InprocConduit::WriteBy(ThreadRank thread, void* DataPtr, int DataSize, int tag)
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
			std::cerr<<"Error, tag resued!"<<std::endl;
			LCK2.unlock();
			exit(1);
		}
		if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
		{
			// has buff and tag not exist, do real write

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

            if(m_srcBuffPoolWaitlist.find(tag) != m_srcBuffPoolWaitlist.end())
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing writeby ptr...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
#ifdef USE_DEBUG_ASSERT
				assert(m_srcBuffPoolWaitlist[tag]==1);
#endif
				m_srcBuffPoolWaitlist.erase(tag);
				tmp_buffinfo->dataPtr = DataPtr;
				m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] = 1;
				LCK2.unlock();
				m_srcNewBuffInsertedCond.notify_all();

				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].wait(LCK3,
						[=](){return m_srcBuffDataReadFlag[tmp_idx] == 1;});
				LCK3.unlock();

				// release buff
                LCK2.lock();
                m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                tmp_buffinfo->dataPtr= nullptr;
                tmp_buffinfo->callingReadThreadCount=0;
                tmp_buffinfo->callingWriteThreadCount=0;
                tmp_buffinfo->reduceBuffsizeSensor=0;
                m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
                tmp_buffinfo->buffIdx = -1;
                m_srcAvailableBuffCount++;
                m_srcBuffPool.erase(tag);
                m_srcBuffAvailableCond.notify_one();
                LCK2.unlock();

			}
			else
			{
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing writeby small msg...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing writeby...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
					LCK2.unlock();
					m_srcNewBuffInsertedCond.notify_all();

					std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_idx]);
					// do real data transfer
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_srcBuffDataWrittenFlag[tmp_idx] =1;
					LCK3.unlock();
					m_srcBuffDataWrittenCond[tmp_idx].notify_one();
				}
			}
			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
			//      m_numSrcLocalThreads-1));
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
			std::cerr<<"Error, tag resued!"<<std::endl;
			LCK2.unlock();
			exit(1);
		}
		if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
		{
			// has buff now, can do the real write

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

			if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
			{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby ptr...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
#ifdef USE_DEBUG_ASSERT
				assert(m_dstBuffPoolWaitlist[tag] ==1);
#endif
				m_dstBuffPoolWaitlist.erase(tag);
				tmp_buffinfo->dataPtr = DataPtr;
				m_dstBuffDataWrittenFlag[tmp_idx] = 1;
				LCK2.unlock();
				m_dstNewBuffInsertedCond.notify_all();

				// wait reader finish data copy
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
				m_dstBuffDataReadCond[tmp_idx].wait(LCK3,
						[=](){return m_dstBuffDataReadFlag[tmp_idx] == 1;});
				LCK3.unlock();

				// release buff
                LCK2.lock();
                m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
                tmp_buffinfo->dataPtr= nullptr;
                tmp_buffinfo->callingReadThreadCount=0;
                tmp_buffinfo->callingWriteThreadCount=0;
                tmp_buffinfo->reduceBuffsizeSensor=0;
                m_dstBuffIdx.push_back(tmp_buffinfo->buffIdx);
                tmp_buffinfo->buffIdx = -1;
                m_dstAvailableBuffCount++;
                m_dstBuffPool.erase(tag);
                m_dstBuffAvailableCond.notify_one();
                LCK2.unlock();
			}
			else
			{
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby small msg...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffDataWrittenFlag[tmp_idx] =1;
					LCK2.unlock();
					m_dstNewBuffInsertedCond.notify_all();
				}
				else
				{
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing writeby...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
					LCK2.unlock();
					m_dstNewBuffInsertedCond.notify_all();

					// get access lock for this buffer to write data
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_idx]);
					// do real data transfer
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffDataWrittenFlag[tmp_idx] =1;
					LCK3.unlock();
					m_dstBuffDataWrittenCond[tmp_idx].notify_one();
				}
			}

			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK4(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,
			//      m_numDstLocalThreads-1));
			m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
			m_writebyFinishCond.notify_all();

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish writeby!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			return 0;
		}
	}// end dsttask
	else
	{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// end WriteBy()


// Actually, it's same as other mode of writeby_finish operation.
void InprocConduit::WriteBy_Finish(int tag)
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

}// end WriteBy_Finish



}// end namespace iUtc

