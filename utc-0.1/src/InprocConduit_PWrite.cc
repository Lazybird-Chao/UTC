#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include "../include/TaskUtilities.h"


namespace iUtc{

/*
 *
 */
int InprocConduit::PWrite(void *DataPtr, DataSize_t DataSize, int tag){
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    // current calling thread's belonging task id
    static thread_local int myTaskid = -1;
    static thread_local int myThreadRank = -1;
    static thread_local int myLocalRank = -1;
    if(myTaskid == -1)
    {
        myTaskid = TaskManager::getCurrentTaskId();
        myThreadRank = TaskManager::getCurrentThreadRankinTask();
        myLocalRank = TaskManager::getCurrentThreadRankInLocal();
        m_srcBuffQueue->setThreadId(myLocalRank);
        //m_srcInnerMsgQueue->setThreadId(myLocalRank);
        //m_dstBuffQueue->setThreadId(myLocalRank);
        //m_dstInnerMsgQueue->setThreadId(myLocalRank);

    }

    if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        if(m_numSrcLocalThreads ==1)
        {
        	// only one local thread for the task
            // get avilable msg buff
        	MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop();
            if(tmp_buffptr == nullptr){
                std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                exit(1);
            }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwrite ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif        
        	tmp_buffptr->dataPtr = DataPtr;
            tmp_buffptr->dataSize = DataSize;
            tmp_buffptr->usingPtr = true;
            tmp_buffptr->msgTag = tag;
            m_srcUsingPtrFinishFlag[myLocalRank].store(0);
            tmp_buffptr->safeRelease = &m_srcUsingPtrFinishFlag[myLocalRank];
        	// push msg to buffqueue for reader to read
        	if(m_srcBuffQueue->push(tmp_buffptr)){
        		std::cerr<<"ERROR, potential write timeout!"<<std::endl;
        		exit(1);
        	}
        	// as use ptr for data transfer, need wait for reader finish
        	while(m_srcUsingPtrFinishFlag[myLocalRank].load()==0){
        		_mm_pause();
        	}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        	
        }
        else{
        	// multi local threads in task
        	if(myThreadRank == m_srcOpTokenFlag[myThreadRank])
            {
                // check if it's current thread rank's turn to do r/w

                // reset next token's latch here
                int next_thread = (m_srcOpTokenFlag[myThreadRank]+1) % m_numSrcLocalThreads;
                m_srcOpThreadLatch[next_thread]->reset(1);
                // do msg r/w
                MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop();
                if(tmp_buffptr == nullptr){
                    std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                    exit(1);
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Pwrite ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                tmp_buffptr->dataPtr = DataPtr;
                tmp_buffptr->dataSize = DataSize;
                tmp_buffptr->usingPtr = true;
                tmp_buffptr->msgTag = tag;
                m_srcUsingPtrFinishFlag[myLocalRank].store(0);
                tmp_buffptr->safeRelease = &m_srcUsingPtrFinishFlag[myLocalRank];
                // push msg to buffqueue for reader to read
                if(m_srcBuffQueue->push(tmp_buffptr)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }
                // as use ptr for data transfer, need wait for reader finish
	        	while(m_srcUsingPtrFinishFlag[myLocalRank].load() == 0){
	        		_mm_pause();
	        	}

                // wake up other waiting thread
                m_srcOpThreadLatch[m_srcOpTokenFlag[myThreadRank]]->count_down();
                m_srcOpTokenFlag[myThreadRank] = next_thread;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            }
            else{
                // not the op thread, just wait the op thread finish
                int do_thread =  m_srcOpTokenFlag[myThreadRank];  //the op thread's rank
                if(!m_srcOpThreadLatch[do_thread]->try_wait()){
                	m_srcOpThreadLatch[do_thread]->wait();         // wait on associated latch
                }
                // wake up when do_thread finish, and update token flag to next value
                m_srcOpTokenFlag[myThreadRank] = (m_srcOpTokenFlag[myThreadRank]+1)%m_numSrcLocalThreads; 
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

            }
        }
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
            MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop();
            if(tmp_buffptr == nullptr){
                std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                exit(1);
            }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            tmp_buffptr->dataPtr = DataPtr;
            tmp_buffptr->dataSize = DataSize;
            tmp_buffptr->usingPtr = true;
            tmp_buffptr->msgTag = tag;
            m_dstUsingPtrFinishFlag[myLocalRank].store(0);
            tmp_buffptr->safeRelease = &m_dstUsingPtrFinishFlag[myLocalRank];
            if(m_dstBuffQueue->push(tmp_buffptr)){
                std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                exit(1);
            }
            // as use ptr for data transfer, need wait for reader finish
        	while(m_dstUsingPtrFinishFlag[myLocalRank].load()==0){
        		_mm_pause();
        		//__asm__ __volatile__ ("pause":::"memory");
        		//asm volatile("pause":::"memocy");
        	}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif                
        }
        else{
            if(myThreadRank == m_dstOpTokenFlag[myThreadRank])
            {
                int next_thread = (m_dstOpTokenFlag[myThreadRank]+1) % m_numDstLocalThreads;
                m_dstOpThreadLatch[next_thread]->reset(1);
                MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop();
                if(tmp_buffptr == nullptr){
                    std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                    exit(1);
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Pwrite ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                tmp_buffptr->dataPtr = DataPtr;
                tmp_buffptr->dataSize = DataSize;
                tmp_buffptr->usingPtr = true;
                tmp_buffptr->msgTag = tag;
                m_dstUsingPtrFinishFlag[myLocalRank].store(0);
                tmp_buffptr->safeRelease = &m_dstUsingPtrFinishFlag[myLocalRank];
                if(m_dstBuffQueue->push(tmp_buffptr)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }
                // as use ptr for data transfer, need wait for reader finish
	        	while(m_dstUsingPtrFinishFlag[myLocalRank] == 0){
	        		_mm_pause();
	        	}

                m_dstOpThreadLatch[m_dstOpTokenFlag[myThreadRank]]->count_down();
                m_dstOpTokenFlag[myThreadRank] = next_thread;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            }
            else{
                int do_thread =  m_dstOpTokenFlag[myThreadRank];  //the op thread's rank
                if(!m_dstOpThreadLatch[do_thread]->try_wait()){
                	m_dstOpThreadLatch[do_thread]->wait();         // wait on associated latch
                }
                // wake up when do_thread finish, and update token flag to next value
                m_dstOpTokenFlag[myThreadRank] = (m_dstOpTokenFlag[myThreadRank]+1)%m_numDstLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif            
            }

        }
    } //end dst
    else{
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;
} // end BWrite()


}


