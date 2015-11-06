#include "Conduit.h"
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

                // last thread release this buffer
                assert(m_dstBuffPool.find(tag) != m_dstBuffPool.end());
                assert(m_dstBuffPool[tag]->callingReadThreadCount == 1);
                std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[m_dstBuffPool[tag]->buffIdx]);
                while(m_dstBuffPool[tag]->safeReleaseAfterRead == false)
				{
					m_dstBuffSafeReleaseCond.wait(LCK3);

				}
                LCK3.unlock();
                std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
                BuffInfo *tmp_buff = m_dstBuffPool[tag];
                assert(tmp_buff->dataPtr!=nullptr);
                if(tmp_buff->isBuffered)
                	free(tmp_buff->dataPtr);
                tmp_buff->dataPtr = nullptr;
                m_dstBuffDataWrittenFlag[tmp_buff->buffIdx] =0;
                m_dstBuffDataReadFlag[tmp_buff->buffIdx] =0;
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
            BuffInfo* tmp_buffinfo;
            std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
            if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
            {
                // tag not exist, means writer haven't come yet
                // add this tag to buffpoolwaitlist
                assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
                m_dstBuffPoolWaitlist[tag] = 1;
                // this useful for Pwrite, as he may wait for a reader come
                /*m_dstBuffPoolWaitlistCond.notify_one();*/
                // go on waiting for this tag-buffer come from writer
                std::cv_status w_ret= std::cv_status::no_timeout;
                while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
                {
                    w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                   if(w_ret == std::cv_status::timeout)
                   {
                       std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                       LCK2.unlock();
                       exit(1);
                   }
                }

                // wake up when writer comes
                assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
                tmp_buffinfo = m_dstBuffPool[tag];
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
                	// for small msg, do not use two-phase msg copy
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read small msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                	memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                	m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                	LCK3.unlock();
                	m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
                else
                {
                	// for big msg, use two-phase copy
					// go on check if data is ready for copy
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
					while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
					{
						m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
					}
					// wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
					// real data transfer
					memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
					m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
					LCK3.unlock();
					// notify writer that read finish
					m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
            }
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
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
                tmp_buffinfo->callingReadThreadCount =1;
                LCK2.unlock();
                if(DataSize<SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read small msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                	memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                	m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                	LCK3.unlock();
                	m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
                else
                {
					// go on check if data is filled
					std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
					while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
					{
						m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
					}
					// wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
					// real data transfer
					memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
					m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
					LCK3.unlock();
					// notify writer that read finish
					m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
            }

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
                std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
                while(tmp_buffinfo->safeReleaseAfterRead == false)
				{
					m_dstBuffSafeReleaseCond.wait(LCK3);
				}
                LCK3.unlock();
                LCK2.lock();
                assert(tmp_buffinfo->dataPtr!=nullptr);
                if(tmp_buffinfo->isBuffered)
                		free(tmp_buffinfo->dataPtr);
                tmp_buffinfo->dataPtr = nullptr;
                m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =0;
                m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =0;
                m_dstBuffIdx.push_back(tmp_buffinfo->buffIdx);
                delete tmp_buffinfo;
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
                assert(m_srcBuffPool.find(tag) != m_srcBuffPool.end());
                assert(m_srcBuffPool[tag]->callingReadThreadCount == 1);
                std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[m_srcBuffPool[tag]->buffIdx]);
                while(m_srcBuffPool[tag]->safeReleaseAfterRead == false)
				{
					m_srcBuffSafeReleaseCond.wait(LCK3);

				}
                LCK3.unlock();
                // last read thread will release the buff
                std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
                BuffInfo *tmp_buffinfo = m_srcBuffPool[tag];
                assert(tmp_buffinfo->dataPtr!=nullptr);
                if(tmp_buffinfo->isBuffered)
                	free(tmp_buffinfo->dataPtr);
                tmp_buffinfo->dataPtr = nullptr;
                m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =0;
                m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =0;
                m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
                delete tmp_buffinfo;
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
            BuffInfo* tmp_buffinfo;
            // check if tag is in buffpool
            std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
            if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
            {
                // tag not exist, means writer haven't come yet
                // add this tag to buffpoolwaitlist
                assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
                m_srcBuffPoolWaitlist[tag]=1;
                /*m_srcBuffPoolWaitlistCond.notify_one();*/
                // go one wait for the msg come
                bool w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT),
                                           [=](){return m_srcBuffPool.find(tag) !=m_srcBuffPool.end(); });
                if(w_ret == false)
                {
                    std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                    LCK2.unlock();
                    exit(1);
                }
                //wake up when msg comes
                assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
                tmp_buffinfo = m_srcBuffPool[tag];
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
                tmp_buffinfo->callingReadThreadCount = 1;
                LCK2.unlock();
                if(DataSize < SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read small msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
    				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
                	memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                	m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
                	LCK3.unlock();
                	m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
                else
                {
                	// wait for data if ready to copy
				   std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				   while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
				   {
					   m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
				   }
				   // wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
					// real data transfer
					memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
					m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
					LCK3.unlock();
					// notify writer that read finish
					m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
            }
            else
            {
                // writer already comes
                tmp_buffinfo = m_srcBuffPool[tag];
                assert(tmp_buffinfo->callingReadThreadCount == 0);
                assert(tmp_buffinfo->callingWriteThreadCount >0);
                tmp_buffinfo->callingReadThreadCount = 1;
                LCK2.unlock();
                if(DataSize < SMALL_MESSAGE_CUTOFF)
                {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read small msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
    				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
    				memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
    				m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
    				LCK3.unlock();
    				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
                else
                {
					// go on check if data is filled
					std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
					while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
					{
						m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
					}
					// wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
					// real data transfer
					memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
					m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
					LCK3.unlock();
					// notify writer that read finish
					m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
                }
            }

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
                std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
                while(tmp_buffinfo->safeReleaseAfterRead == false)
				{
					m_srcBuffSafeReleaseCond.wait(LCK3);
				}
                LCK3.unlock();
                LCK2.lock();
                assert(tmp_buffinfo->dataPtr!=nullptr);
                if(tmp_buffinfo->isBuffered)
                	free(tmp_buffinfo->dataPtr);
                tmp_buffinfo->dataPtr = nullptr;
                m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =0;
                m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =0;
                m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
                delete tmp_buffinfo;
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
        BuffInfo *tmp_buffinfo;
        std::unique_lock<std::mutex> LCK2(m_dstBuffManagerMutex);
        if(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
        {
            // no tag in buffpool, means the message hasn't come yet
            // add tag to buffpoolwaitlist
            assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
            m_dstBuffPoolWaitlist.insert(std::pair<MessageTag,int>(tag, 1));
            // signal writer that reader has come
            /*m_dstBuffPoolWaitlistCond.notify_one();*/
            // go on waiting for this tag-buffer come from writer
            std::cv_status w_ret= std::cv_status::no_timeout;
            while(m_dstBuffPool.find(tag) == m_dstBuffPool.end())
            {
                w_ret = m_dstNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT));
                if(w_ret == std::cv_status::timeout)
                {
                  std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                  LCK2.unlock();
                  exit(1);
                }
            }
            // wake up when writer comes
            assert(m_dstBuffPoolWaitlist.find(tag) == m_dstBuffPoolWaitlist.end());
            tmp_buffinfo = m_dstBuffPool[tag];
            assert(tmp_buffinfo->callingReadThreadCount == 0);
            assert(tmp_buffinfo->callingWriteThreadCount >0);
            tmp_buffinfo->callingReadThreadCount =1;
            LCK2.unlock();
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby small msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
    			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
    			memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
    			m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
    			LCK3.unlock();
    			m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
            else
            {
				// wait for writer fill in the data
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
				{
					m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
				}
				// wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
				// real data transfer
				memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
				m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
				LCK3.unlock();
				// notify reader copy complete
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
        }
        else
        {
            // tag is already in buffpool
            tmp_buffinfo = m_dstBuffPool[tag];
            assert(tmp_buffinfo->callingReadThreadCount == 0);
            assert(tmp_buffinfo->callingWriteThreadCount >0);
            tmp_buffinfo->callingReadThreadCount = 1;
            LCK2.unlock();
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby small msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
    			std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
    			memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
    			m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
    			LCK3.unlock();
    			m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
            else
            {
				// go on check if data is filled
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				while(m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
				{
					m_dstBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
				}
				// data is filled in
	#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" doing readby...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
	#endif
				// real data transfer
				memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
				m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
				LCK3.unlock();
				// notify reader copy complete
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
        }

        // as only this assigned thread do read, so release buff after read
        std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
        while(tmp_buffinfo->safeReleaseAfterRead == false)
        {
            m_dstBuffSafeReleaseCond.wait(LCK3);
        }
        LCK3.unlock();
        LCK2.lock();
        assert(tmp_buffinfo->dataPtr!=nullptr);
        if(tmp_buffinfo->isBuffered)
        	free(tmp_buffinfo->dataPtr);
        tmp_buffinfo->dataPtr = nullptr;
        m_dstBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =0;
        m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =0;
        m_dstBuffIdx.push_back(tmp_buffinfo->buffIdx);
        delete tmp_buffinfo;
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
        BuffInfo *tmp_buffinfo;
        std::unique_lock<std::mutex> LCK2(m_srcBuffManagerMutex);
        if(m_srcBuffPool.find(tag) == m_srcBuffPool.end())
        {
            // tag not exist, means writer haven't come yet
            assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
            m_srcBuffPoolWaitlist.insert(std::pair<MessageTag, int>(tag, 1));
            /*m_srcBuffPoolWaitlistCond.notify_one();*/
            // go on waiting for the msg come
            bool w_ret = m_srcNewBuffInsertedCond.wait_for(LCK2, std::chrono::seconds(TIME_OUT),
                                       [=](){return m_srcBuffPool.find(tag) !=m_srcBuffPool.end();});
            if(w_ret == false)
            {
                std::cout<<"Error, reader wait time out!"<<"src-thread "<<myThreadRank<<std::endl;
                LCK2.unlock();
                exit(1);
            }
            //wake up when msg comes
            assert(m_srcBuffPoolWaitlist.find(tag)==m_srcBuffPoolWaitlist.end());
            tmp_buffinfo = m_srcBuffPool[tag];
            assert(tmp_buffinfo->callingReadThreadCount == 0);
            assert(tmp_buffinfo->callingWriteThreadCount >0);
            tmp_buffinfo->callingReadThreadCount = 1;
            LCK2.unlock();

            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby small msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
    			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
    			memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
				m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] = 1;
				LCK3.unlock();
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
            else
            {
			   // wait for writer come and fill in the data
			   std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
			   while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
			   {
				   m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
			   }
			   // wake up when the data is filled
	#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
	#endif
				// real data transfer
				memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
				m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] = 1;
				LCK3.unlock();
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
        }
        else
        {
            // writer already comes
            tmp_buffinfo = m_srcBuffPool[tag];
            assert(tmp_buffinfo->callingReadThreadCount == 0);
            assert(tmp_buffinfo->callingWriteThreadCount >0);
            tmp_buffinfo->callingReadThreadCount = 1;
            LCK2.unlock();
            if(DataSize<SMALL_MESSAGE_CUTOFF)
            {
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby small msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
    			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
    			memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
                m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] = 1;
                LCK3.unlock();
                m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
            else
            {
				// go on check if data is filled
				std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
				while(m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] ==0)
				{
					m_srcBuffDataWrittenCond[tmp_buffinfo->buffIdx].wait(LCK3);
				}
				// wake up when the data is filled
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing readby...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
				// real data transfer
				memcpy(DataPtr,tmp_buffinfo->dataPtr, DataSize);
				m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] = 1;
				LCK3.unlock();
				m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
            }
        }

        // need release buff after read
        std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
        while(tmp_buffinfo->safeReleaseAfterRead == false)
		{
			m_srcBuffSafeReleaseCond.wait(LCK2);
		}
        LCK3.unlock();
        LCK2.lock();
        assert(tmp_buffinfo->dataPtr!=nullptr);
        if(tmp_buffinfo->isBuffered)
        	free(tmp_buffinfo->dataPtr);
        tmp_buffinfo->dataPtr = nullptr;
        m_srcBuffDataWrittenFlag[tmp_buffinfo->buffIdx] =0;
        m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =0;
        m_srcBuffIdx.push_back(tmp_buffinfo->buffIdx);
        delete tmp_buffinfo;
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
