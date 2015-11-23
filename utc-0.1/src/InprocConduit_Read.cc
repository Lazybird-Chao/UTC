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
 *  Blocking conduit read operation
 */
int InprocConduit::Read(void *DataPtr, DataSize_t DataSize, int tag)
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
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
        if(m_numSrcLocalThreads == 1)
        {
            // only one local thread, no need for op lock
            // check if msg tag is in the pool
            BuffInfo* tmp_buffinfo;
            std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
            if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
            {
                // tag not exist, means writer haven't come yet
                // add this tag to buffpoolwaitlist
#ifdef USE_DEBUG_ASSERT
                assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
                m_dstBuffPoolWaitlist[tag] = 1;
                // go on waiting for this tag-buffer come from writer
                std::cv_status w_ret= std::cv_status::no_timeout;
                while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
                {
                    w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                    if(w_ret == std::cv_status::timeout)
                    {
                        std::cerr<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                        LCK2.unlock();
                        exit(1);
                    }
                }
                // wake up when writer comes
#ifdef USE_DEBUG_ASSERT
                assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
                tmp_buffinfo = m_dstBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
                if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
                {
                    // use inter buffer for write
                    if(DataSize < SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_ASSERT
                        assert(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==1);
#endif
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                        while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                        {
                            m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                        }
                        // real data transfer
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                        LCK3.unlock();
                    }

                    // release buff
                    LCK2.lock();
                    m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
                    // use address for write
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                    // notify writer that read finish
                    m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
            }// end for tag not in the pool
            else
            {
                // find tag, means writer already comes
                /*
                 *  here we assume that there is not tag reuse problem, every msg has
                 *  a unique tag.
                 *  if has tag reuse, a slow thread that has not finish read would cause
                 *  a fast thread find the tag exist, but it's a stale msg. This is not
                 *  easy to deal with.
                 *
                 *  TODO: change buffpool organization, allowing msg with same tag, and
                 *  msg of same tag will be arranged as a queue.
                 *  Same as MPI does, in-order recv in same (tag, comm) tupple, different
                 *  tag can do out-of-order recv
                 */
                tmp_buffinfo = m_dstBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
                if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
                {
                    // use inter buffer for write
                    if(DataSize < SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_ASSERT
                        assert(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==1);
#endif
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                        while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                        {
                            m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                        }
                        // real data transfer
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                        //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                        LCK3.unlock();
                    }

                    // release buff
                    LCK2.lock();
                    m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
                    // use address for write
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                    // notify writer that read finish
                    m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }

            }//end for tag in the pool

        }// end for one thread
        else
        {
            // there are several local threads
            // get op check lock
            std::unique_lock<std::mutex> LCK1(m_srcOpCheckMutex);
            int counteridx = m_srcOpRotateCounterIdx[myThreadRank];
            m_srcOpRotateCounter[counteridx]++;
            if(m_srcOpRotateCounter[counteridx] > 1)
            {
                // a late coming thread
#ifdef USE_DEBUG_ASSERT
                assert(m_srcOpRotateCounter[counteridx] <= m_numSrcLocalThreads);
#endif
                while(m_srcOpRotateFinishFlag[counteridx] ==0)
                {
                    m_srcOpFinishCond.wait(LCK1);
                }

                // wake up after real read finish
                m_srcOpRotateFinishFlag[counteridx]++;
                if(m_srcOpRotateFinishFlag[counteridx] == m_numSrcLocalThreads)
                {
                    // last leaving thread
                    m_srcOpRotateCounter[counteridx]=0;
                    m_srcOpRotateFinishFlag[counteridx]=0;
                    m_srcOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
                    m_srcAvailableNoFinishedOpCount++;
                    m_srcAvailableNoFinishedOpCond.notify_one();
                    LCK1.unlock();
                }
                else
                {
                    m_srcOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
                    LCK1.unlock();
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
                return 0;
            }// end for late thread
            else
            {
                // the first coming thread
                while(m_srcAvailableNoFinishedOpCount==0)
                {
                    m_srcAvailableNoFinishedOpCond.wait(LCK1);
                }
                m_srcAvailableNoFinishedOpCount--;
                LCK1.unlock();

                BuffInfo* tmp_buffinfo;
                std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
                if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
                {
                    // tag not exist, means writer haven't come yet
                    // add this tag to buffpoolwaitlist
#ifdef USE_DEBUG_ASSERT
                    assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
                    m_dstBuffPoolWaitlist[tag] = 1;
                }
                // go on waiting for this tag-buffer come from writer
                std::cv_status w_ret= std::cv_status::no_timeout;
                while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
                {
                    w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                    if(w_ret == std::cv_status::timeout)
                    {
                        std::cerr<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                        LCK2.unlock();
                        exit(1);
                    }
                }
                // wake up when writer comes
#ifdef USE_DEBUG_ASSERT
                assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
                tmp_buffinfo = m_dstBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
                if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
                {
                    // use inter buffer for write
                    if(DataSize < SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_ASSERT
                        assert(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]==1);
#endif
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                        while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                        {
                            m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                        }
                        // real data transfer
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                        //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                        LCK3.unlock();
                    }

                    // release buff
                    LCK2.lock();
                    m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
                    // use address for write
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                    // notify writer that read finish
                    m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }

                // first thread finish read, need change opfinishflag
                LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                assert(m_srcOpRotateFinishFlag[counteridx] ==0);
#endif
                m_srcOpRotateFinishFlag[counteridx]++;
                m_srcOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
                m_srcOpFinishCond.notify_all();
                LCK1.unlock();
            }// end for first thread

        }// end for several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
        return 0;

    }// end src read
    else if(myTaskid == m_dstId)
    {
        // dst task calling read()
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1)
        {
            // only one local thread, no need for op lock
            // check if msg tag is in the pool
            BuffInfo* tmp_buffinfo;
            std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
            if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
            {
                // tag not exist, means writer haven't come yet
                // add this tag to buffpoolwaitlist
#ifdef USE_DEBUG_ASSERT
                assert(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end());
#endif
                m_srcBuffPoolWaitlist[tag] = 1;
            }
            // go on waiting for this tag-buffer come from writer
            std::cv_status w_ret= std::cv_status::no_timeout;
            while(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
            {
                w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                if(w_ret == std::cv_status::timeout)
                {
                    std::cerr<<"Error, reader wait time out!"<<"dst-thread "<<myThreadRank<<std::endl;
                    LCK2.unlock();
                    exit(1);
                }
            }
            // wake up when writer comes
#ifdef USE_DEBUG_ASSERT
            assert(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end());
#endif
            tmp_buffinfo = m_srcBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
            assert(tmp_buffinfo->callingReadThreadCount == 0);
            assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
            tmp_buffinfo->callingReadThreadCount =1;
            LCK2.unlock();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
            if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
            {
                // use inter buffer for write
                if(DataSize < SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_ASSERT
                    assert(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]==1);
#endif
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                }
                else
                {
                    std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
                    while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                    {
                        m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                    }
                    // real data transfer
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                    LCK3.unlock();
                }

                // release buff
                LCK2.lock();
                m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
                // use address for write
                memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                // notify writer that read finish
                m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }

        }// end for one thread
        else
        {
            // there are several local threads
            // get op check lock
            std::unique_lock<std::mutex> LCK1(m_dstOpCheckMutex);
            int counteridx = m_dstOpRotateCounterIdx[myThreadRank];
            m_dstOpRotateCounter[counteridx]++;
            if(m_dstOpRotateCounter[counteridx] > 1)
            {
                // a late coming thread
#ifdef USE_DEBUG_ASSERT
                assert(m_dstOpRotateCounter[counteridx] <= m_numDstLocalThreads);
#endif
                while(m_dstOpRotateFinishFlag[counteridx] ==0)
                {
                    m_dstOpFinishCond.wait(LCK1);
                }

                // wake up after real read finish
                m_dstOpRotateFinishFlag[counteridx]++;
                if(m_dstOpRotateFinishFlag[counteridx] == m_numDstLocalThreads)
                {
                    // last leaving thread
                    m_dstOpRotateCounter[counteridx]=0;
                    m_dstOpRotateFinishFlag[counteridx]=0;
                    m_dstOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
                    m_dstAvailableNoFinishedOpCount++;
                    m_dstAvailableNoFinishedOpCond.notify_one();
                    LCK1.unlock();
                }
                else
                {
                    m_dstOpRotateCounterIdx[myThreadRank]= (counteridx +1)%(m_noFinishedOpCapacity +1);
                    LCK1.unlock();
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
                return 0;
            }// end for late thread
            else
            {
                // the first coming thread
                while(m_dstAvailableNoFinishedOpCount==0)
                {
                    m_dstAvailableNoFinishedOpCond.wait(LCK1);
                }
                m_dstAvailableNoFinishedOpCount--;
                LCK1.unlock();

                BuffInfo* tmp_buffinfo;
                std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
                if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
                {
                    // tag not exist, means writer haven't come yet
                    // add this tag to buffpoolwaitlist
#ifdef USE_DEBUG_ASSERT
                    assert(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end());
#endif
                    m_srcBuffPoolWaitlist[tag] = 1;
                }
                // go on waiting for this tag-buffer come from writer
                std::cv_status w_ret= std::cv_status::no_timeout;
                while(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
                {
                    w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                    if(w_ret == std::cv_status::timeout)
                    {
                        std::cerr<<"Error, reader wait time out!"<<"dst-thread "<<myThreadRank<<std::endl;
                        LCK2.unlock();
                        exit(1);
                    }
                }
                // wake up when writer comes
#ifdef USE_DEBUG_ASSERT
                assert(m_srcBuffPoolWaitlist.find(tag) == m_srcBuffPoolWaitlist.end());
#endif
                tmp_buffinfo = m_srcBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
                if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
                {
                    // use inter buffer for write
                    if(DataSize < SMALL_MESSAGE_CUTOFF)
                    {
#ifdef USE_DEBUG_ASSERT
                        assert(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]==1);
#endif
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
                        while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                        {
                            m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                        }
                        // real data transfer
                        memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                        //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                        LCK3.unlock();
                    }

                    // release buff
                    LCK2.lock();
                    m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
                    //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
                    // use address for write
                    memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                    m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                    // notify writer that read finish
                    m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }

                // first thread finish read, need chang opfinishflag
                LCK1.lock();
#ifdef USE_DEBUG_ASSERT
                assert(m_dstOpRotateFinishFlag[counteridx] ==0);
#endif
                m_dstOpRotateFinishFlag[counteridx]++;
                m_dstOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
                m_dstOpFinishCond.notify_all();
                LCK1.unlock();
            }// end for first thread

        }// end for several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        return 0;

    }// end dst read
    else
    {
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;

}// end read


int InprocConduit::ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
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
            std::cerr<<"Error, thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
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
        BuffInfo *tmp_buffinfo;
        std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
        if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
        {
            // no tag in buffpool, means the message hasn't come yet
            // add tag to buffpoolwaitlist
#ifdef USE_DEBUG_ASSERT
            assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
            m_dstBuffPoolWaitlist.insert(std::pair<MessageTag_t,int>(tag, 1));
        }

        // go on waiting for this tag-buffer come from writer
        bool w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT),
                      [=](){return m_dstBuffPool.find(tag) !=m_dstBuffPool.end();});
        if(!w_ret)
        {
          std::cerr<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
          LCK2.unlock();
          exit(1);
        }

        // wake up when writer comes
#ifdef USE_DEBUG_ASSERT
        assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
#endif
        tmp_buffinfo = m_dstBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
        assert(tmp_buffinfo->callingReadThreadCount == 0);
        assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
        tmp_buffinfo->callingReadThreadCount =1;
        LCK2.unlock();

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
        if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
        {
            // use inter buffer for write
            if(DataSize < SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_ASSERT
                assert(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]==1);
#endif
                memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
            }
            else
            {
                std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                {
                    m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                }
                // real data transfer
                memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                LCK3.unlock();
            }

            // release buff
            LCK2.lock();
            m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
            //m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
            // use address for write
            memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
            m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
            // notify writer that read finish
            m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
        }

        if(m_numSrcLocalThreads>1)
        {
			// record this op in finish set
			std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
			m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
			m_readbyFinishCond.notify_all();
        }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish readby:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
    }
    else if(myTaskid == m_dstId)
    {
        BuffInfo *tmp_buffinfo;
        std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
        if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
        {
            // tag not exist, means writer haven't come yet
#ifdef USE_DEBUG_ASSERT
            assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
#endif
            m_srcBuffPoolWaitlist.insert(std::pair<MessageTag_t, int>(tag, 1));
        }

        // go on waiting for the msg come
        bool w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT),
                                   [=](){return m_srcBuffPool.find(tag) !=m_srcBuffPool.end();});
        if(w_ret == false)
        {
            std::cerr<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
            LCK2.unlock();
            exit(1);
        }
        //wake up when msg comes
#ifdef USE_DEBUG_ASSERT
        assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
#endif
        tmp_buffinfo = m_srcBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
        assert(tmp_buffinfo->callingReadThreadCount == 0);
        assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
        tmp_buffinfo->callingReadThreadCount = 1;
        LCK2.unlock();

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        if(tmp_buffinfo->bufferPtr == tmp_buffinfo->dataPtr)
        {
            // use inter buffer for write
            if(DataSize < SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_ASSERT
                assert(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]==1);
#endif
                memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
            }
            else
            {
                std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
                while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
                {
                    m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
                }
                // real data transfer
                memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                LCK3.unlock();
            }

            // release buff
            LCK2.lock();
            m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx]=0;
            //m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx]=0;
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
            // use address for write
            memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
            m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
            // notify writer that read finish
            m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
        }

        if(m_numDstLocalThreads>1)
        {
			// record this op in finish set
			std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
			m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
			m_readbyFinishCond.notify_all();
        }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish readby:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
    }
    else
    {
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;

}// end ReadBy()

void InprocConduit::ReadBy_Finish(int tag)
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
        // we use (tag<<LOG_MAX_TASKS)+myTaskid as the internal tag here,
        // as for src and dst we use one finish set, so to need differentiate
        // the srctag and dsttag

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




}// end namespace iUtc
