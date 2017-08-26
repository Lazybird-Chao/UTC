#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
#include "../include/TaskUtilities.h"
#include "TimerUtilities.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>



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
        //m_srcBuffQueue->setThreadId(myLocalRank);
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
        	MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
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
        	if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
        		std::cerr<<"ERROR, potential write timeout!"<<std::endl;
        		exit(1);
        	}
        	// as use ptr for data transfer, need wait for reader finish
        	long _counter=0;
        	while(m_srcUsingPtrFinishFlag[myLocalRank].load()==0){
        		_counter++;
				/*if(_counter<USE_PAUSE)
					_mm_pause();
				else if(_counter<USE_SHORT_SLEEP){
					__asm__ __volatile__ ("pause" ::: "memory");
					std::this_thread::yield();
				}
				else if(_counter<USE_LONG_SLEEP)
					nanosleep(&SHORT_PERIOD, nullptr);
				else
					nanosleep(&LONG_PERIOD, nullptr);
					*/
        		spinWait(_counter);
        	}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        	
        }
        else{
        	// multi local threads in task
        	int idx = m_srcOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_srcOpThreadAvailable.size());
			//assert(m_srcOpThreadFinish[idx]->load() != 0);
#endif
			if(m_srcOpThreadAvailable[idx]->compare_exchange_strong(isavailable,1))
			{

                // do msg r/w
                MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
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
                if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }
                // as use ptr for data transfer, need wait for reader finish
                long _counter=0;
	        	while(m_srcUsingPtrFinishFlag[myLocalRank].load() == 0){
	        		_counter++;
					/*if(_counter<USE_PAUSE)
						_mm_pause();
					else if(_counter<USE_SHORT_SLEEP){
						__asm__ __volatile__ ("pause" ::: "memory");
						std::this_thread::yield();
					}
					else if(_counter<USE_LONG_SLEEP)
						nanosleep(&SHORT_PERIOD, nullptr);
					else
						nanosleep(&LONG_PERIOD, nullptr);
						*/
	        		spinWait(_counter);
	        	}

	        	// wake up other waiting thread

				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					m_srcOpThreadFinishLatch[idx]->count_down();
				}
				else{

					m_srcOpThreadFinish[idx]->store(0);
				}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish Pwrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
            }
            else{
            	// not the op thread, just wait the op thread finish
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					boost::latch* tmp_latch = m_srcOpThreadFinishLatch[idx];
					if(!tmp_latch->try_wait()){
						tmp_latch->wait();         // wait on associated latch
					}
				}
				else{
					long _counter=0;
					while(m_srcOpThreadFinish[idx]->load() != 0){
						_counter++;
						/*if(_counter<USE_PAUSE)
							_mm_pause();
						else if(_counter<USE_SHORT_SLEEP){
							__asm__ __volatile__ ("pause" ::: "memory");
							std::this_thread::yield();
						}
						else if(_counter<USE_LONG_SLEEP)
							nanosleep(&SHORT_PERIOD, nullptr);
						else
							nanosleep(&LONG_PERIOD, nullptr);
							*/
						spinWait(_counter);
					}
				}

				int nthreads = m_numSrcLocalThreads-1;
				while(1){
					int oldvalue = m_srcOpThreadAvailable[idx]->load();
					if(oldvalue == nthreads){
						m_srcOpThreadAvailable[idx]->store(0);
						if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
							m_srcOpThreadFinishLatch[idx]->reset(1);

						}
						else{
							m_srcOpThreadFinish[idx]->store(1);
						}
						break;
					}
					if(m_srcOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
						break;
				}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit Pwrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

            }
			m_srcOpTokenFlag[myLocalRank]++;
			m_srcOpTokenFlag[myLocalRank]=m_srcOpTokenFlag[myLocalRank]%m_nOps2;
        }
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
            MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(myLocalRank);
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
            if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
                std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                exit(1);
            }
            // as use ptr for data transfer, need wait for reader finish
            long _counter=0;
        	while(m_dstUsingPtrFinishFlag[myLocalRank].load()==0){
        		_counter++;
				/*if(_counter<USE_PAUSE)
					_mm_pause();
				else if(_counter<USE_SHORT_SLEEP){
					__asm__ __volatile__ ("pause" ::: "memory");
					std::this_thread::yield();
				}
				else if(_counter<USE_LONG_SLEEP)
					nanosleep(&SHORT_PERIOD, nullptr);
				else
					nanosleep(&LONG_PERIOD, nullptr);
					*/
        		spinWait(_counter);
        	}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif                
        }
        else{
        	// multiple threads
			int idx = m_dstOpTokenFlag[myLocalRank];
			int isavailable = 0;
			if(m_dstOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1))
			{

                MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(myLocalRank);
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
                if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
                    std::cerr<<"ERROR, potential write timeout!"<<std::endl;
                    exit(1);
                }
                // as use ptr for data transfer, need wait for reader finish
                long _counter=0;
	        	while(m_dstUsingPtrFinishFlag[myLocalRank] == 0){
	        		_counter++;
					/*if(_counter<USE_PAUSE)
						_mm_pause();
					else if(_counter<USE_SHORT_SLEEP){
						__asm__ __volatile__ ("pause" ::: "memory");
						std::this_thread::yield();
					}
					else if(_counter<USE_LONG_SLEEP)
						nanosleep(&SHORT_PERIOD, nullptr);
					else
						nanosleep(&LONG_PERIOD, nullptr);
						*/
	        		spinWait(_counter);
	        	}


				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					m_dstOpThreadFinishLatch[idx]->count_down();
				}
				else{
					m_dstOpThreadFinish[idx]->store(0);
				}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish Pwrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
            }
            else{
            	boost::latch* temp_latch = m_dstOpThreadFinishLatch[idx];
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!temp_latch->try_wait()){
						temp_latch->wait();
					}
				}
				else{
					long _counter=0;
					while(m_dstOpThreadFinish[idx]->load() !=0){
						_counter++;
						/*if(_counter<USE_PAUSE)
							_mm_pause();
						else if(_counter<USE_SHORT_SLEEP){
							__asm__ __volatile__ ("pause" ::: "memory");
							std::this_thread::yield();
						}
						else if(_counter<USE_LONG_SLEEP)
							nanosleep(&SHORT_PERIOD, nullptr);
						else
							nanosleep(&LONG_PERIOD, nullptr);
							*/
						spinWait(_counter);
					}
				}

				int nthreads = m_numDstLocalThreads-1;
				while(1){
					int oldvalue = m_dstOpThreadAvailable[idx]->load();
					if(oldvalue == nthreads){
						m_dstOpThreadAvailable[idx]->store(0);
						if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
							m_dstOpThreadFinishLatch[idx]->reset(1);
						}
						else{
							m_dstOpThreadFinish[idx]->store(1);
						}

						break;
					}
					if(m_dstOpThreadAvailable[idx]->compare_exchange_strong(oldvalue,oldvalue+1))
						break;
				}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit Pwrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif            
            }
			m_dstOpTokenFlag[myLocalRank]++;
			m_dstOpTokenFlag[myLocalRank]=m_dstOpTokenFlag[myLocalRank]%m_nOps2;
        }
    } //end dst
    else{
        std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;
} // end PWrite()


int InprocConduit::PWriteBy(ThreadRank_t thread, void *DataPtr, DataSize_t DataSize, int tag){
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
        *m_threadOstream<<"thread "<<myThreadRank<<" call Pwriteby..."<<std::endl;
#endif
    if(myThreadRank != thread){
    	if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cerr<<"Error, pwriteby thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}
    	else if(TaskManager::getCurrentTask()->isLocal(thread) == false){
			std::cerr<<"Error, pwriteby thread rank "<<myThreadRank<<" is not on main process!"<<std::endl;
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
    if(myTaskid == m_srcId){
    	MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(myLocalRank);
		if(tmp_buffptr == nullptr){
			std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
			exit(1);
		}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing PwriteBy ...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		tmp_buffptr->dataPtr = DataPtr;
		tmp_buffptr->dataSize = DataSize;
		tmp_buffptr->usingPtr = true;
		tmp_buffptr->msgTag = tag;
		m_srcUsingPtrFinishFlag[myLocalRank].store(0);
		tmp_buffptr->safeRelease = &m_srcUsingPtrFinishFlag[myLocalRank];
		// push msg to buffqueue for reader to read
		if(m_srcBuffQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential write timeout!"<<std::endl;
			exit(1);
		}
		// as use ptr for data transfer, need wait for reader finish
		long _counter=0;
		while(m_srcUsingPtrFinishFlag[myLocalRank].load()==0){
			_counter++;
			/*if(_counter<USE_PAUSE)
				_mm_pause();
			else if(_counter<USE_SHORT_SLEEP){
				__asm__ __volatile__ ("pause" ::: "memory");
				std::this_thread::yield();
			}
			else if(_counter<USE_LONG_SLEEP)
				nanosleep(&SHORT_PERIOD, nullptr);
			else
				nanosleep(&LONG_PERIOD, nullptr);
				*/
			spinWait(_counter);
		}
#ifdef ENALBE_OPBY_FINISH
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish PwriteBy!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing PwriteBy ...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		tmp_buffptr->dataPtr = DataPtr;
		tmp_buffptr->dataSize = DataSize;
		tmp_buffptr->usingPtr = true;
		tmp_buffptr->msgTag = tag;
		m_dstUsingPtrFinishFlag[myLocalRank].store(0);
		tmp_buffptr->safeRelease = &m_dstUsingPtrFinishFlag[myLocalRank];
		if(m_dstBuffQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential write timeout!"<<std::endl;
			exit(1);
		}
		// as use ptr for data transfer, need wait for reader finish
		long _counter=0;
		while(m_dstUsingPtrFinishFlag[myLocalRank].load()==0){
			_counter++;
			/*if(_counter<USE_PAUSE)
				_mm_pause();
			else if(_counter<USE_SHORT_SLEEP){
				__asm__ __volatile__ ("pause" ::: "memory");
				std::this_thread::yield();
			}
			else if(_counter<USE_LONG_SLEEP)
				nanosleep(&SHORT_PERIOD, nullptr);
			else
				nanosleep(&LONG_PERIOD, nullptr);
				*/
			spinWait(_counter);
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish PwriteBy!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    	exit(1);
    }
    return 0;

}//end PwriteBy


#ifdef ENABLE_OPBY_FINISH
void InprocConduit::PWriteBy_Finish(int tag)
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
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait Pwriteby!"<<std::endl;
#endif
		return;
	}
	else if(myTaskid == m_dstId && m_numDstLocalThreads == 1)
	{
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" exit wait Pwriteby!"<<std::endl;
#endif
		return;
	}

    std::unique_lock<std::mutex> LCK1(m_writebyFinishMutex);
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" wait for Pwriteby..."<<std::endl;
#endif
    while(m_writebyFinishSet.find((tag<<LOG_MAX_TASKS)+myTaskid) ==
            m_writebyFinishSet.end())
    {
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

        m_writebyFinishSet.erase((tag<<LOG_MAX_TASKS)+myTaskid);
    }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"thread "<<myThreadRank<<" finish wait Pwriteby!"<<std::endl;
#endif

    return;

}// end PWriteBy_Finish
#endif


}


