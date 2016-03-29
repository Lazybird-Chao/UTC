#include "InprocConduit.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "../include/TaskUtilities.h"

#include <cassert>
//#include <utility>
#include <thread>


namespace iUtc{

int InprocConduit::AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag){
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
    }

    if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1){
			// only one local thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

			AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
			args->DataPtr= DataPtr;
			args->DataSize= DataSize;
			args->tag = tag;
			args->WorkType = 2; // type-write

			m_srcAsyncWriteFinishSet[tag] = std::promise<void>();
			m_srcAsyncWorkQueue->push(args, myLocalRank);
			if(m_srcAsyncWorkerOn.load() == false){
				// no async worker, create one
				m_srcAsyncWorkerOn.store(true);
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}
			else{
				// async worker is there
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
		}
		else{
			// multiple threads in task
			int idx = m_srcAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_srcAsyncOpThreadAvailable.size());
			//assert(m_srcAsyncOpThreadFinish[idx]->load() != 0);
#endif
			// first coming thread
			if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_srcAsyncOpThreadAvailable.size()-1);
#endif
				// push next item to vector
				m_srcAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_srcAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
				args->DataPtr= DataPtr;
				args->DataSize= DataSize;
				args->tag = tag;
				args->WorkType = 2; // type-write

				m_srcAsyncWriteFinishSet[tag] = std::promise<void>();
				m_srcAsyncWorkQueue->push(args, myLocalRank);
				if(m_srcAsyncWorkerOn.load() == false){
					// no async worker, create one
					m_srcAsyncWorkerOn.store(true);
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}

				m_srcAsyncOpThreadFinish[idx]->store(0);

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			}
			else{
				// other threads wait
				long _counter=0;
				while(m_srcAsyncOpThreadFinish[idx]->load() !=0){
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

				int nthreads = m_numSrcLocalThreads-1;
				while(1){
					int oldvalue = m_srcAsyncOpThreadAvailable[idx]->load();
					if(oldvalue==nthreads){
						delete m_srcAsyncOpThreadAvailable[idx];
						m_srcAsyncOpThreadAvailable[idx] = nullptr;
						delete m_srcAsyncOpThreadFinish[idx];
						m_srcAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
						break;
				}

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			}
			m_srcAsyncOpTokenFlag[myLocalRank]++;
		}// end several threads
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call AsyncWrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		if(m_numDstLocalThreads == 1){
			// only one local thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
			args->DataPtr= DataPtr;
			args->DataSize= DataSize;
			args->tag = tag;
			args->WorkType = 2; // type-write

			m_dstAsyncWriteFinishSet[tag] = std::promise<void>();
			m_dstAsyncWorkQueue->push(args, myLocalRank);

			if(m_dstAsyncWorkerOn.load() == false){
				m_dstAsyncWorkerOn.store(true);
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
		}
		else{
			// multi threads in task
			int idx = m_dstAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
			if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
				m_dstAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_dstAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
				args->DataPtr= DataPtr;
				args->DataSize= DataSize;
				args->tag = tag;
				args->WorkType = 2; // type-write

				m_dstAsyncWriteFinishSet[tag] = std::promise<void>();
				m_dstAsyncWorkQueue->push(args, myLocalRank);

				if(m_dstAsyncWorkerOn.load() == false){
					m_dstAsyncWorkerOn.store(true);
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}

				m_dstAsyncOpThreadFinish[idx]->store(0);

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			}
			else{
				// other threads wait
				long _counter=0;
				while(m_dstAsyncOpThreadFinish[idx]->load() !=0){
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
				int nthreads = m_numDstLocalThreads-1;
				while(1){
					int oldvalue = m_dstAsyncOpThreadAvailable[idx]->load();
					if(oldvalue ==nthreads){
						delete m_dstAsyncOpThreadAvailable[idx];
						m_dstAsyncOpThreadAvailable[idx]=nullptr;
						delete m_dstAsyncOpThreadFinish[idx];
						m_dstAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_weak(oldvalue,oldvalue+1))
						break;
				}

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit AsyncWrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
			}
			m_dstAsyncOpTokenFlag[myLocalRank]++;
		}//end several threads

    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
    }

    return 0;
}// end async_write()



void InprocConduit::asyncWorkerImpl(int myTaskid){
#ifdef USE_DEBUG_ASSERT
	assert(m_srcAsyncWorkerOn == true || m_dstAsyncWorkerOn == true);
#endif
#ifdef USE_DEBUG_LOG
	m_asyncWorkerCount++;
#endif
	AsyncWorkArgs_t *m_args=nullptr;
	int closeWorkerCounter = 0;
	if(myTaskid == m_srcId){
		while(1){
			m_args = m_srcAsyncWorkQueue->try_pop(0);
			//std::cout<<m_args<<std::endl;
			if(m_args == nullptr){
				// nothing to pop, no item in workqueue
				closeWorkerCounter++;
				if(closeWorkerCounter > m_closeWorkerCountMax){
					m_srcAsyncWorkerOn.store(false);
					break;
				}
				std::this_thread::yield();
			}
			else{
				// get item from queue, mean there's async op request
				//std::cout<<m_args->WorkType<<std::endl;
				closeWorkerCounter=0;
				if(m_args->WorkType == 1){
					threadReadImpl(m_args->DataPtr, m_args->DataSize, m_args->tag, myTaskid);
					m_srcAsyncReadFinishSet[m_args->tag].set_value();
				}
				else if(m_args->WorkType == 2){
					threadWriteImpl(m_args->DataPtr, m_args->DataSize, m_args->tag, myTaskid);
					//std::cout<<"here1"<<std::endl;
					m_srcAsyncWriteFinishSet[m_args->tag].set_value();
				}
				else{
					std::cerr<<"Error, undefined worktype for asyncworker!!!"<<std::endl;
					exit(1);
				}
				free(m_args);
			}
		}// end worker
	}// end src
	else{
		while(1){
			m_args = m_dstAsyncWorkQueue->try_pop(0);
			if(m_args == nullptr){
				// nothing to pop, no item in workqueue
				closeWorkerCounter++;
				if(closeWorkerCounter > m_closeWorkerCountMax){
					m_dstAsyncWorkerOn.store(false);
					break;
				}
				std::this_thread::yield();
			}
			else{
				// get item from queue, mean there's async op request
				closeWorkerCounter=0;
				if(m_args->WorkType == 1){
					threadReadImpl(m_args->DataPtr, m_args->DataSize, m_args->tag, myTaskid);
					m_dstAsyncReadFinishSet[m_args->tag].set_value();
				}
				else if(m_args->WorkType == 2){
					threadWriteImpl(m_args->DataPtr, m_args->DataSize, m_args->tag,myTaskid);
					m_dstAsyncWriteFinishSet[m_args->tag].set_value();
				}
				else{
					std::cerr<<"Error, undefined worktype for asyncworker!!!"<<std::endl;
					exit(1);
				}
				free(m_args);
			}
		}// end worker
	}// end dst
	return;
}

int InprocConduit::threadWriteImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid){
	if(myTaskid == m_srcId){
		MsgInfo_t *tmp_buffptr = m_srcInnerMsgQueue->pop(m_numSrcLocalThreads);
		if(tmp_buffptr == nullptr){
            std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
            exit(1);
        }
        if(DataSize <= CONDUIT_BUFFER_SIZE){
    		// small message, no need malloc space
    	   	tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
    	   	memcpy(tmp_buffptr->dataPtr, DataPtr, DataSize);
    	   	tmp_buffptr->usingPtr = false;
    	}
    	else{
    		// big msg, using ptr for transfer
    		tmp_buffptr->dataPtr= DataPtr;
    		tmp_buffptr->usingPtr = true;
    		tmp_buffptr->safeRelease = &m_srcUsingPtrFinishFlag[m_numSrcLocalThreads];
    		m_srcUsingPtrFinishFlag[m_numSrcLocalThreads].store(0);
    	}
       	tmp_buffptr->dataSize = DataSize;
        tmp_buffptr->msgTag = tag;
    	// push msg to buffqueue for reader to read
    	if(m_srcBuffQueue->push(tmp_buffptr, m_numSrcLocalThreads)){
    		std::cerr<<"ERROR, potential write timeout!"<<std::endl;
    		exit(1);
    	}
    	if(DataSize > CONDUIT_BUFFER_SIZE){
    		// big msg using ptr, need wait reader finish
    		long _counter=0;
    		while(m_srcUsingPtrFinishFlag[m_numSrcLocalThreads].load() == 0){
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
	}
	else{
		MsgInfo_t *tmp_buffptr = m_dstInnerMsgQueue->pop(m_numDstLocalThreads);
        if(tmp_buffptr == nullptr){
            std::cerr<<"ERROR, potential get buff timeout!"<<std::endl;
            exit(1);
        }
        if(DataSize <= CONDUIT_BUFFER_SIZE){
    		// small message, no need malloc space
    	   	tmp_buffptr->dataPtr = tmp_buffptr->smallDataBuff;
    	   	memcpy(tmp_buffptr->dataPtr, DataPtr, DataSize);
    	   	tmp_buffptr->usingPtr = false;
    	}
    	else{
    		// big msg, using ptr for transfer
    		tmp_buffptr->dataPtr= DataPtr;
    		tmp_buffptr->usingPtr = true;
    		tmp_buffptr->safeRelease = &m_dstUsingPtrFinishFlag[m_numDstLocalThreads];
    		m_dstUsingPtrFinishFlag[m_numDstLocalThreads].store(0);
    	}
        tmp_buffptr->dataSize = DataSize;
        tmp_buffptr->msgTag = tag;
        if(m_dstBuffQueue->push(tmp_buffptr, m_numDstLocalThreads)){
            std::cerr<<"ERROR, potential write timeout!"<<std::endl;
            exit(1);
        }
        if(DataSize > CONDUIT_BUFFER_SIZE){
        	long _counter=0;
        	while(m_dstUsingPtrFinishFlag[m_numDstLocalThreads].load() == 0){
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

	}
	return 0;
}

void InprocConduit::AsyncWrite_Finish(int tag){
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
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

	if(myTaskid == m_srcId){
		// src
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" wait for AsyncWrite..."<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1)
		{
			std::future<void> fut = m_srcAsyncWriteFinishSet[tag].get_future();
			fut.wait();
			m_srcAsyncWriteFinishSet.erase(tag);
		}
		else{
			int idx = m_srcAsyncOpTokenFlag[myLocalRank];
			int isavailable =0;
			if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
				// push next item to vector
				m_srcAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_srcAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

				std::future<void> fut = m_srcAsyncWriteFinishSet[tag].get_future();
				//*m_threadOstream<<"here"<<std::endl;
				fut.wait();
				m_srcAsyncWriteFinishSet.erase(tag);

				m_srcAsyncOpThreadFinish[idx]->store(0);
			}
			else{
				long _counter=0;
				while(m_srcAsyncOpThreadFinish[idx]->load() != 0){
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
				int nthreads = m_numSrcLocalThreads-1;
				while(1){
					int oldvalue = m_srcAsyncOpThreadAvailable[idx]->load();
					if(oldvalue==nthreads){
						delete m_srcAsyncOpThreadAvailable[idx];
						m_srcAsyncOpThreadAvailable[idx] = nullptr;
						delete m_srcAsyncOpThreadFinish[idx];
						m_srcAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
						break;
				}
			}
			m_srcAsyncOpTokenFlag[myLocalRank]++;
		}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
	}
	else{
		// dst
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" wait for AsyncWrite..."<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
        	// only one local thread
			std::future<void> fut = m_dstAsyncWriteFinishSet[tag].get_future();
			fut.wait();
			m_dstAsyncWriteFinishSet.erase(tag);
        }
        else{
        	int idx = m_dstAsyncOpTokenFlag[myLocalRank];
			int isavailable =0;
			if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
				// push next item to vector
				m_dstAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_dstAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

				std::future<void> fut = m_dstAsyncWriteFinishSet[tag].get_future();

				fut.wait();
				m_dstAsyncWriteFinishSet.erase(tag);

				m_dstAsyncOpThreadFinish[idx]->store(0);
			}
        	else{
        		long _counter=0;
				while(m_dstAsyncOpThreadFinish[idx]->load() != 0){
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
				int nthreads = m_numDstLocalThreads-1;
				while(1){
					int oldvalue = m_dstAsyncOpThreadAvailable[idx]->load();
					if(oldvalue ==nthreads){
						delete m_dstAsyncOpThreadAvailable[idx];
						m_dstAsyncOpThreadAvailable[idx]=nullptr;
						delete m_dstAsyncOpThreadFinish[idx];
						m_dstAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_weak(oldvalue,oldvalue+1))
						break;
				}
        	}
			m_dstAsyncOpTokenFlag[myLocalRank]++;
        }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
	}
	return;

}

int InprocConduit::AsyncRead(void* DataPtr, DataSize_t DataSize, int tag){
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
    if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1){
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
    		AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
    		args->DataPtr = DataPtr;
    		args->DataSize = DataSize;
    		args->tag = tag;
    		args->WorkType = 1; // type-read
    		m_srcAsyncReadFinishSet[tag] = std::promise<void>();
    		m_srcAsyncWorkQueue->push(args, myLocalRank);
    		if(m_srcAsyncWorkerOn.load() == false){
    			m_srcAsyncWorkerOn.store(true);
    			std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
    		}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncRead:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		}
		else{
			// multiple threads in task
			int idx = m_srcAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_srcAsyncOpThreadAvailable.size());
			//assert(m_srcAsyncOpThreadFinish[idx]->load() != 0);
#endif
			// first coming thread
			if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_srcAsyncOpThreadAvailable.size()-1);
#endif
				// push next item to vector
				m_srcAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_srcAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
    			AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
        		args->DataPtr = DataPtr;
        		args->DataSize = DataSize;
        		args->tag = tag;
        		args->WorkType = 1; // type-read
        		m_srcAsyncReadFinishSet[tag] = std::promise<void>();
        		m_srcAsyncWorkQueue->push(args, myLocalRank);
        		if(m_srcAsyncWorkerOn.load() == false){
        			m_srcAsyncWorkerOn.store(true);
        			std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
        		}

        		m_srcAsyncOpThreadFinish[idx]->store(0);

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncRead:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			}
			else{
				long _counter=0;
				while(m_srcAsyncOpThreadFinish[idx]->load() !=0){
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

				int nthreads = m_numSrcLocalThreads-1;
				while(1){
					int oldvalue = m_srcAsyncOpThreadAvailable[idx]->load();
					if(oldvalue==nthreads){
						delete m_srcAsyncOpThreadAvailable[idx];
						m_srcAsyncOpThreadAvailable[idx] = nullptr;
						delete m_srcAsyncOpThreadFinish[idx];
						m_srcAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
						break;
				}
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			}
			m_srcAsyncOpTokenFlag[myLocalRank]++;
		}
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
			args->DataPtr= DataPtr;
			args->DataSize= DataSize;
			args->tag = tag;
			args->WorkType = 1; // type-read

			m_dstAsyncReadFinishSet[tag] = std::promise<void>();
			m_dstAsyncWorkQueue->push(args, myLocalRank);

			if(m_dstAsyncWorkerOn.load() == false){
				m_dstAsyncWorkerOn.store(true);
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncRead:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        }
        else{
        	int idx = m_dstAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
			if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
				m_dstAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_dstAsyncOpThreadFinish.push_back(new std::atomic<int>(1));
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
				AsyncWorkArgs_t *args = new AsyncWorkArgs_t;
				args->DataPtr= DataPtr;
				args->DataSize= DataSize;
				args->tag = tag;
				args->WorkType = 1; // type-read

				m_dstAsyncReadFinishSet[tag] = std::promise<void>();
				m_dstAsyncWorkQueue->push(args, myLocalRank);

				if(m_dstAsyncWorkerOn.load() == false){
					m_dstAsyncWorkerOn.store(true);
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}

				m_dstAsyncOpThreadFinish[idx]->store(0);

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncRead:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        	}
        	else{
        		long _counter=0;
				while(m_dstAsyncOpThreadFinish[idx]->load() !=0){
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
				int nthreads = m_numDstLocalThreads-1;
				while(1){
					int oldvalue = m_dstAsyncOpThreadAvailable[idx]->load();
					if(oldvalue ==nthreads){
						delete m_dstAsyncOpThreadAvailable[idx];
						m_dstAsyncOpThreadAvailable[idx]=nullptr;
						delete m_dstAsyncOpThreadFinish[idx];
						m_dstAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_weak(oldvalue,oldvalue+1))
						break;
				}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        	}
			m_dstAsyncOpTokenFlag[myLocalRank]++;
        }
    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
    	exit(1);
    }
    return 0;
}


int InprocConduit::threadReadImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid){
	if(myTaskid == m_srcId){
		MsgInfo_t *tmp_buffptr;
		/*tmp_buffptr = m_dstBuffQueue->pop(m_numSrcLocalThreads);
		if(tmp_buffptr == nullptr){
			// can't get new item form queue, means no items in queue that wirter writes to
			std::cerr<<"ERROR, potential read timeout!"<<std::endl;
			exit(1);
		}
		if(tmp_buffptr->msgTag != tag){
			m_dstBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
			if(m_dstBuffMap.find(tag) != m_dstBuffMap.end()){
				// find wanted msg in map
				tmp_buffptr = m_dstBuffMap[tag];
				m_dstBuffMap.erase(tag);
			}
			else{
				// not in map, it can be only still in queue
				while((tmp_buffptr = m_dstBuffQueue->pop(m_numSrcLocalThreads))!=nullptr){
					if(tmp_buffptr->msgTag == tag)
						break;
					else
						m_dstBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
				}
				if(tmp_buffptr == nullptr){
					// can't get new item form queue, means no items in queue that wirter writes to
					std::cerr<<"ERROR, potential read timeout!"<<std::endl;
					exit(1);
				}
			}
			// find wanted msg
		}*/
		long _counter=0;
		while(1){
			tmp_buffptr = m_dstBuffQueue->try_pop(m_numSrcLocalThreads);
			if(tmp_buffptr!=nullptr && tmp_buffptr->msgTag == tag)
				break;
			else{
				if(tmp_buffptr!=nullptr)
					m_dstBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
				if(m_dstBuffMap.find(tag) != m_dstBuffMap.end()){
					// find msg in map
					tmp_buffptr = m_dstBuffMap[tag];
					m_dstBuffMap.erase(tag);
					break;
				}
			}
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

		if(tmp_buffptr->usingPtr){
			// use ptr for transfer
#ifdef USE_DEBUG_ASSERT
			assert((tmp_buffptr->safeRelease)->load()==0);
#endif
			char *p_s = (char*)tmp_buffptr->dataPtr;
			char *p_d = (char*)DataPtr;
			long tmp_size = DataSize;
			for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
			{
				memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
				p_d=p_d+INPROC_COPY_THRESHHOLD;
				p_s=p_s+INPROC_COPY_THRESHHOLD;
				tmp_size = tmp_size - INPROC_COPY_THRESHHOLD;
			}
			memcpy(p_d, p_s, tmp_size);
			// tell writer, here finish read, he can go
			(tmp_buffptr->safeRelease)->store(1);
		}
		else{
			// use intermediate buffer
			char *p_s = (char*)tmp_buffptr->dataPtr;
			char *p_d = (char*)DataPtr;
			long tmp_size = DataSize;
			for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
			{
				memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
				p_d=p_d+INPROC_COPY_THRESHHOLD;
				p_s=p_s+INPROC_COPY_THRESHHOLD;
				tmp_size -= INPROC_COPY_THRESHHOLD;
			}
			memcpy(p_d, p_s, tmp_size);
			if(DataSize > CONDUIT_BUFFER_SIZE)
				// big msg space is malloced, need free after read
				free(tmp_buffptr->dataPtr);
		}
		//
		tmp_buffptr->dataSize = 0;
		tmp_buffptr->usingPtr = false;
		tmp_buffptr->msgTag = -1;
		tmp_buffptr->dataPtr = nullptr;
		tmp_buffptr->safeRelease = nullptr;
		// return this buffer to dst's inner msg queue
		if(m_dstInnerMsgQueue->push(tmp_buffptr,m_numSrcLocalThreads)){
			std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
			exit(1);
		}
	}// end src
	else{
		// dst
		MsgInfo_t *tmp_buffptr;
		/*tmp_buffptr = m_srcBuffQueue->pop(m_numDstLocalThreads);
		if(tmp_buffptr == nullptr){
			// can't get new item form queue, means no items in queue that wirter writes to
			std::cerr<<"ERROR, potential read timeout!"<<std::endl;
			exit(1);
		}
		if(tmp_buffptr->msgTag != tag){
			m_srcBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
			if(m_srcBuffMap.find(tag) != m_srcBuffMap.end()){
				// find wanted msg in map
				tmp_buffptr = m_srcBuffMap[tag];
				m_srcBuffMap.erase(tag);
			}
			else{
				// not in map, it can be only still in queue
				while((tmp_buffptr = m_srcBuffQueue->pop(m_numDstLocalThreads))!=nullptr){
					if(tmp_buffptr->msgTag == tag)
						break;
					else
						m_srcBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
				}
				if(tmp_buffptr == nullptr){
					// can't get new item form queue, means no items in queue that wirter writes to
					std::cerr<<"ERROR, potential read timeout!"<<std::endl;
					exit(1);
				}
			}
			// find wanted msg
		}*/
		long _counter=0;
		while(1){
			tmp_buffptr = m_srcBuffQueue->try_pop(m_numDstLocalThreads);
			if(tmp_buffptr!=nullptr && tmp_buffptr->msgTag == tag)
				break;
			else{
				if(tmp_buffptr!=nullptr)
					m_srcBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
				if(m_srcBuffMap.find(tag) != m_srcBuffMap.end()){
					// find msg in map
					tmp_buffptr = m_srcBuffMap[tag];
					m_srcBuffMap.erase(tag);
					break;
				}

			}
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

		if(tmp_buffptr->usingPtr){
			// use ptr for transfer
#ifdef USE_DEBUG_ASSERT
			assert((tmp_buffptr->safeRelease)->load()==0);
#endif
			char *p_s = (char*)tmp_buffptr->dataPtr;
			char *p_d = (char*)DataPtr;
			long tmp_size = DataSize;
			for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
			{
				memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
				p_d=p_d+INPROC_COPY_THRESHHOLD;
				p_s=p_s+INPROC_COPY_THRESHHOLD;
				tmp_size -= INPROC_COPY_THRESHHOLD;
			}
			memcpy(p_d, p_s, tmp_size);
			// tell writer, here finish read, he can go
			(tmp_buffptr->safeRelease)->store(1);
		}
		else{
			// use intermediate buffer
			char *p_s = (char*)tmp_buffptr->dataPtr;
			char *p_d = (char*)DataPtr;
			long tmp_size = DataSize;
			for(int i=0; i<(DataSize+INPROC_COPY_THRESHHOLD-1)/INPROC_COPY_THRESHHOLD -1; i++)
			{
				memcpy(p_d, p_s, INPROC_COPY_THRESHHOLD);
				p_d=p_d+INPROC_COPY_THRESHHOLD;
				p_s=p_s+INPROC_COPY_THRESHHOLD;
				tmp_size -= INPROC_COPY_THRESHHOLD;
			}
			memcpy(p_d, p_s, tmp_size);
			if(DataSize > CONDUIT_BUFFER_SIZE)
				// big msg space is malloced, need free after read
				free(tmp_buffptr->dataPtr);
		}
		//
		tmp_buffptr->dataSize = 0;
		tmp_buffptr->usingPtr = false;
		tmp_buffptr->msgTag = -1;
		tmp_buffptr->dataPtr = nullptr;
		tmp_buffptr->safeRelease = nullptr;
		// return this buffer to dst's inner msg queue
		if(m_srcInnerMsgQueue->push(tmp_buffptr,m_numDstLocalThreads)){
			std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
			exit(1);
		}

	}// end dst
	return 0;
}


void InprocConduit::AsyncRead_Finish(int tag){
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
	}
	if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

	if(myTaskid == m_srcId){
		// src
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" wait for AsyncRead..."<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1)
		{
			std::future<void> fut = m_srcAsyncReadFinishSet[tag].get_future();
			fut.wait();
			m_srcAsyncReadFinishSet.erase(tag);
		}
		else{
			// multiple threads in task
			int idx = m_srcAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_srcAsyncOpThreadAvailable.size());
			//assert(m_srcAsyncOpThreadFinish[idx]->load() != 0);
#endif
			// first coming thread
			if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
#ifdef USE_DEBUG_ASSERT
			assert(idx == m_srcAsyncOpThreadAvailable.size()-1);
#endif
				// push next item to vector
				m_srcAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_srcAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

				std::future<void> fut = m_srcAsyncReadFinishSet[tag].get_future();
				fut.wait();
				m_srcAsyncReadFinishSet.erase(tag);

				m_srcAsyncOpThreadFinish[idx]->store(0);

			}
			else{
				long _counter=0;
				while(m_srcAsyncOpThreadFinish[idx]->load() !=0){
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

				int nthreads = m_numSrcLocalThreads-1;
				while(1){
					int oldvalue = m_srcAsyncOpThreadAvailable[idx]->load();
					if(oldvalue==nthreads){
						delete m_srcAsyncOpThreadAvailable[idx];
						m_srcAsyncOpThreadAvailable[idx] = nullptr;
						delete m_srcAsyncOpThreadFinish[idx];
						m_srcAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_srcAsyncOpThreadAvailable[idx]->compare_exchange_strong(oldvalue, oldvalue+1))
						break;
				}
			}
			m_srcAsyncOpTokenFlag[myLocalRank]++;
		}
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" finish wait AsyncRead!"<<std::endl;
#endif
	}
	else{
		// dst
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" wait for AsyncRead..."<<std::endl;
#endif
        if(m_numDstLocalThreads == 1){
        	// only one local thread
			std::future<void> fut = m_dstAsyncReadFinishSet[tag].get_future();
			fut.wait();
			m_dstAsyncReadFinishSet.erase(tag);
        }
        else{
        	int idx = m_dstAsyncOpTokenFlag[myLocalRank];
			int isavailable = 0;
			if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
				m_dstAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
				m_dstAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

				std::future<void> fut = m_dstAsyncReadFinishSet[tag].get_future();
				fut.wait();
				m_dstAsyncReadFinishSet.erase(tag);

				m_dstAsyncOpThreadFinish[idx]->store(0);
        	}
        	else{
        		long _counter=0;
				while(m_dstAsyncOpThreadFinish[idx]->load() !=0){
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
				int nthreads = m_numDstLocalThreads-1;
				while(1){
					int oldvalue = m_dstAsyncOpThreadAvailable[idx]->load();
					if(oldvalue ==nthreads){
						delete m_dstAsyncOpThreadAvailable[idx];
						m_dstAsyncOpThreadAvailable[idx]=nullptr;
						delete m_dstAsyncOpThreadFinish[idx];
						m_dstAsyncOpThreadFinish[idx]=nullptr;
						break;
					}
					if(m_dstAsyncOpThreadAvailable[idx]->compare_exchange_weak(oldvalue,oldvalue+1))
						break;
				}
        	}
			m_dstAsyncOpTokenFlag[myLocalRank]++;
        }
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish wait AsyncRead!"<<std::endl;
#endif
	}
	return;
}


}// end namespace iUtc


