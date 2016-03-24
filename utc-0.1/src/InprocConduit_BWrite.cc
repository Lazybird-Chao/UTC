#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
#include "UtcBasics.h"
#include "../include/TaskUtilities.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>



namespace iUtc{


/*
 *
 */
int InprocConduit::BWrite(void *DataPtr, DataSize_t DataSize, int tag){
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
        //m_srcBuffQueue->setThreadId(myLocalRank);
    }

    if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        if(m_numSrcLocalThreads ==1)
        {
        	// only one local thread for the task
            // get avilable msg buff
        	MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
            if(tmp_buffptr == nullptr){
                std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                exit(1);
            }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif        
        	if(DataSize <= CONDUIT_BUFFER_SIZE)
        		// small message, no need malloc space
        	   tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;	
        	else
                // big msg, malloc space dynamiclly
        		tmp_buffptr->dataPtr= (char*)malloc(DataSize);
        	char *p_d = (char*)tmp_buffptr->dataPtr;
        	char *p_s = (char*)DataPtr;
        	long tmp_size = DataSize;
        	for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
        	{
        		memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
        		p_d=p_d+INPROC_COPY_THRESHHOLD;
        		p_s=p_s+INPROC_COPY_THRESHHOLD;
        		tmp_size -= INPROC_COPY_THRESHHOLD;
        	}
        	memcpy(p_d, p_s, tmp_size);

            tmp_buffptr->dataSize = DataSize;
            tmp_buffptr->usingPtr = false;
            tmp_buffptr->msgTag = tag;
        	// push msg to buffqueue for reader to read
        	if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
        		std::cerr<<"ERROR, potential write timeout!"<<std::endl;
        		exit(1);
        	}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Bwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        	
        }
        else{
        	// multi local threads in task
        	if(myLocalRank == m_srcOpTokenFlag[myLocalRank])
            {
                // check if it's current thread rank's turn to do r/w

                // reset next token's latch here
                int next_thread = (m_srcOpTokenFlag[myLocalRank]+1) % m_numSrcLocalThreads;
                m_srcOpThreadLatch[next_thread]->reset(1);
                m_srcOpThreadAtomic[next_thread].store(1);
                // do msg r/w
                MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
                if(tmp_buffptr == nullptr){
                    std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                    exit(1);
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" doing Bwrite ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
                if(DataSize <= CONDUIT_BUFFER_SIZE)
                    // small message, no need malloc space
                   tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;   
                else
                    // big msg, malloc space dynamiclly
                    tmp_buffptr->dataPtr= (char*)malloc(DataSize);

                char *p_d = (char*)tmp_buffptr->dataPtr;
				char *p_s = (char*)DataPtr;
				long tmp_size = DataSize;
				for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
				{
					memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
					p_d=p_d+INPROC_COPY_THRESHHOLD;
					p_s=p_s+INPROC_COPY_THRESHHOLD;
					tmp_size -= INPROC_COPY_THRESHHOLD;
				}
				memcpy(p_d, p_s, tmp_size);

                tmp_buffptr->dataSize = DataSize;
                tmp_buffptr->usingPtr = false;
                tmp_buffptr->msgTag = tag;
                // push msg to buffqueue for reader to read
                if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }

                // wake up other waiting thread
                if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
                	m_srcOpThreadLatch[m_srcOpTokenFlag[myLocalRank]]->count_down();
                else
                	m_srcOpThreadAtomic[m_srcOpTokenFlag[myLocalRank]].store(0);
                m_srcOpTokenFlag[myLocalRank] = next_thread;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Bwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            }
            else{
                // not the op thread, just wait the op thread finish
                int do_thread =  m_srcOpTokenFlag[myLocalRank];  //the op thread's rank
                if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!m_srcOpThreadLatch[do_thread]->try_wait())         // wait on associated latch
					{
						m_srcOpThreadLatch[do_thread]->wait();
					}
                }
                else{
                	long _counter=0;
                	while(m_srcOpThreadAtomic[do_thread].load()!=0){
                		_counter++;
						if(_counter<USE_PAUSE)
							_mm_pause();
						else if(_counter<USE_SHORT_SLEEP){
							__asm__ __volatile__ ("pause" ::: "memory");
							std::this_thread::yield();
						}
						else if(_counter<USE_LONG_SLEEP)
							nanosleep(&SHORT_PERIOD, nullptr);
						else
							nanosleep(&LONG_PERIOD, nullptr);
                	}
                }
                // wake up when do_thread finish, and update token flag to next value
                m_srcOpTokenFlag[myLocalRank] = (m_srcOpTokenFlag[myLocalRank]+1)%m_numSrcLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Bwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

            }
        }
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
            MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(myLocalRank);
            if(tmp_buffptr == nullptr){
                std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                exit(1);
            }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            if(DataSize < CONDUIT_BUFFER_SIZE)
                tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
            else
                tmp_buffptr->dataPtr = (char*)malloc(DataSize);
            char *p_d = (char*)tmp_buffptr->dataPtr;
			char *p_s = (char*)DataPtr;
			long tmp_size = DataSize;
			for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
			{
				memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
				p_d=p_d+INPROC_COPY_THRESHHOLD;
				p_s=p_s+INPROC_COPY_THRESHHOLD;
				tmp_size -= INPROC_COPY_THRESHHOLD;
			}
			memcpy(p_d, p_s, tmp_size);
            tmp_buffptr->dataSize = DataSize;
            tmp_buffptr->usingPtr = false;
            tmp_buffptr->msgTag = tag;
            if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
                std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                exit(1);
            }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Bwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif                
        }
        else{
            if(myLocalRank == m_dstOpTokenFlag[myLocalRank])
            {
                int next_thread = (m_dstOpTokenFlag[myLocalRank]+1) % m_numDstLocalThreads;
                m_dstOpThreadLatch[next_thread]->reset(1);
                m_dstOpThreadAtomic[next_thread].store(1);
                MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(myLocalRank);
                if(tmp_buffptr == nullptr){
                    std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
                    exit(1);
                }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" doing Bwrite ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
                if(DataSize < CONDUIT_BUFFER_SIZE)
                    tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
                else
                    tmp_buffptr->dataPtr = (char*)malloc(DataSize);
                char *p_d = (char*)tmp_buffptr->dataPtr;
				char *p_s = (char*)DataPtr;
				long tmp_size = DataSize;
				for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
				{
					memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
					p_d=p_d+INPROC_COPY_THRESHHOLD;
					p_s=p_s+INPROC_COPY_THRESHHOLD;
					tmp_size -= INPROC_COPY_THRESHHOLD;
				}
				memcpy(p_d, p_s, tmp_size);
                tmp_buffptr->dataSize = DataSize;
                tmp_buffptr->usingPtr = false;
                tmp_buffptr->msgTag = tag;
                if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }

                if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
                	m_dstOpThreadLatch[m_dstOpTokenFlag[myLocalRank]]->count_down();
                else
                	m_dstOpThreadAtomic[m_dstOpTokenFlag[myLocalRank]].store(0);
                m_dstOpTokenFlag[myLocalRank] = next_thread;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Bwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            }
            else{
                int do_thread =  m_dstOpTokenFlag[myLocalRank];  //the op thread's rank
                if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!m_dstOpThreadLatch[do_thread]->try_wait())         // wait on associated latch
					{
						m_dstOpThreadLatch[do_thread]->wait();
					}
                }
                else{
                	long _counter=0;
                	while(m_dstOpThreadAtomic[do_thread].load() !=0){
                		_counter++;
						if(_counter<USE_PAUSE)
							_mm_pause();
						else if(_counter<USE_SHORT_SLEEP){
							__asm__ __volatile__ ("pause" ::: "memory");
							std::this_thread::yield();
						}
						else if(_counter<USE_LONG_SLEEP)
							nanosleep(&SHORT_PERIOD, nullptr);
						else
							nanosleep(&LONG_PERIOD, nullptr);
                	}
                }
                // wake up when do_thread finish, and update token flag to next value
                m_dstOpTokenFlag[myLocalRank] = (m_dstOpTokenFlag[myLocalRank]+1)%m_numDstLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Bwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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



int InprocConduit::BWriteBy(ThreadRank_t thread, void *DataPtr, DataSize_t DataSize, int tag){
#ifdef USE_DEBUG_LOG
    if(!m_threadOstream)
        m_threadOstream = getThreadOstream();
#endif
    static thread_local int myTaskid = -1;
    static thread_local int myThreadRank = -1;
    static thread_local int myLocalRank = -1;
    if(myTaskid == -1)
    {
        myTaskid = TaskManager::getCurrentTaskId();
        myThreadRank = TaskManager::getCurrentThreadRankinTask();
        myLocalRank = TaskManager::getCurrentThreadRankInLocal();
        //*m_threadOstream<<"dst-here"<<myTaskid<<std::endl;
    }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" call Bwriteby..."<<std::endl;
#endif
    if(myThreadRank != thread){
    	if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
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
    if(myTaskid == m_srcId){
    	MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
		if(tmp_buffptr == nullptr){
			std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
			exit(1);
		}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing BwriteBy ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		if(DataSize <= CONDUIT_BUFFER_SIZE)
			// small message, no need malloc space
		   tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
		else
			// big msg, malloc space dynamiclly
			tmp_buffptr->dataPtr= (char*)malloc(DataSize);
		char *p_d = (char*)tmp_buffptr->dataPtr;
		char *p_s = (char*)DataPtr;
		long tmp_size = DataSize;
		for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
		{
			memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
			p_d=p_d+INPROC_COPY_THRESHHOLD;
			p_s=p_s+INPROC_COPY_THRESHHOLD;
			tmp_size -= INPROC_COPY_THRESHHOLD;
		}
		memcpy(p_d, p_s, tmp_size);

		tmp_buffptr->dataSize = DataSize;
		tmp_buffptr->usingPtr = false;
		tmp_buffptr->msgTag = tag;
		// push msg to buffqueue for reader to read
		if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential write timeout!"<<std::endl;
			exit(1);
		}
#ifdef ENABLE_OPBY_FINISH
		if(m_numSrcLocalThreads >1)
		{
			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,m_numSrcLocalThreads-1));
			m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
			m_writebyFinishCond.notify_all();
		}
#endif

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish BwriteBy!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
    }// end src
    else if(myTaskid == m_dstId){
    	MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(myLocalRank);
		if(tmp_buffptr == nullptr){
			std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
			exit(1);
		}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing BwriteBy ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		if(DataSize < CONDUIT_BUFFER_SIZE)
			tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
		else
			tmp_buffptr->dataPtr = (char*)malloc(DataSize);
		char *p_d = (char*)tmp_buffptr->dataPtr;
		char *p_s = (char*)DataPtr;
		long tmp_size = DataSize;
		for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
		{
			memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
			p_d=p_d+INPROC_COPY_THRESHHOLD;
			p_s=p_s+INPROC_COPY_THRESHHOLD;
			tmp_size -= INPROC_COPY_THRESHHOLD;
		}
		memcpy(p_d, p_s, tmp_size);
		tmp_buffptr->dataSize = DataSize;
		tmp_buffptr->usingPtr = false;
		tmp_buffptr->msgTag = tag;
		if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential write timeout!"<<std::endl;
			exit(1);
		}
#ifdef ENABLE_OPBY_FINISH
		if(m_numDstLocalThreads > 1)
		{
			// record this op to readby finish set
			std::lock_guard<std::mutex> LCK(m_writebyFinishMutex);
			//m_writebyFinishSet.insert(std::pair<int, int>((tag<<LOG_MAX_TASKS)+myTaskid,m_numDstLocalThreads-1));
			m_writebyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
			m_writebyFinishCond.notify_all();
		}
#endif
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish BwriteBy!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    	exit(1);
    }
    return 0;

}//end BwriteBy


#ifdef ENABLE_OPBY_FINISH
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
    if(myTaskid == m_srcId && m_numSrcLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait Bwriteby!"<<std::endl;
#endif
		return;
	}
	else if(myTaskid == m_dstId && m_numDstLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait Bwriteby!"<<std::endl;
#endif
		return;
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
#endif





}// end namespace iUtc
