#include "Conduit.h"
#include "TaskManager.h"
#include "Task.h"

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




}// end namespace iUtc
