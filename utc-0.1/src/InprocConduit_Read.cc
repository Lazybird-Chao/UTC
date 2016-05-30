#include "InprocConduit.h"
#include "TaskManager.h"
#include "Task.h"
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
int InprocConduit::Read(void *DataPtr, DataSize_t DataSize, int tag){
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
       	//m_srcBuffQueue->setThreadId(myLocalRank);
    }

    if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1){
			// only one thread in task
			MsgInfo_t *tmp_buffptr;

			/*// fetch one msg from buffer queue
			tmp_buffptr = m_dstBuffQueue->pop(myLocalRank);
			if(tmp_buffptr == nullptr){
				// can't get new item form queue, means no items in queue that wirter writes to
				std::cerr<<"ERROR, potential read timeout!"<<std::endl;
                exit(1);
			}
			if(tmp_buffptr->msgTag != tag){
				// not the wanted msg, so msg either still in queue or be read by other and put in map pool.
				// check the map first, here need ensure only one thread at a time operate on map.
				// so for read, we only allow one thread a time to get msg from queue or check map, even with
				// readyby function, this may cause other thread a little delay in readby function.

				m_dstBuffMap.insert(std::pair<int, MsgInfo_t*>(tmp_buffptr->msgTag, tmp_buffptr));
				if(m_dstBuffMap.find(tag) != m_dstBuffMap.end()){
					// find wanted msg in map
					tmp_buffptr = m_dstBuffMap[tag];
					m_dstBuffMap.erase(tag);
				}
				else{
					// not in map, it can be only still in queue
					while((tmp_buffptr = m_dstBuffQueue->pop(myLocalRank))!=nullptr){
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
				tmp_buffptr = m_dstBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif			
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
				//*m_threadOstream<<tmp_buffptr->msgTag<<"  "<<tmp_buffptr->dataPtr<<std::endl;
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
			if(m_dstInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
				std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
				exit(1);
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		}// end one thread
		else{
			// multiple threads in local
			int idx = m_srcOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_srcOpThreadAvailable.size());
			//assert(m_srcOpThreadFinish[idx]->load() != 0);
#endif
			// first coming thread
			if(m_srcOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1)){
#ifdef USE_DEBUG_ASSERT
			assert(idx = m_srcOpThreadAvailable.size()-1);
#endif

				// doing read op
				MsgInfo_t	*tmp_buffptr;
				long _counter=0;
				while(1){
					tmp_buffptr = m_dstBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
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
				if(m_dstInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
					std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
					exit(1);
				}

				// wake up other threads

				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					m_srcOpThreadFinishLatch[idx]->count_down();
				}
				else{

					m_srcOpThreadFinish[idx]->store(0);
				}

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			}
			else{
				// late coming threads
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					boost::latch *temp_latch = m_srcOpThreadFinishLatch[idx];
					if(!temp_latch->try_wait()){
						temp_latch->wait();
					}
				}
				else{
					long _counter=0;
					while(m_srcOpThreadFinish[idx]->load() !=0){
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			}
			// increase self index
			m_srcOpTokenFlag[myLocalRank]++;
			m_srcOpTokenFlag[myLocalRank]=m_srcOpTokenFlag[myLocalRank]%m_nOps2;

			/*
			if(myLocalRank == m_srcOpTokenFlag[myLocalRank])
			{
				// the right thread's turn do r/w
				int next_thread = (m_srcOpTokenFlag[myLocalRank]+1) % m_numSrcLocalThreads;
				m_srcOpThreadLatch[next_thread]->reset(1);
				m_srcOpThreadAtomic[next_thread].store(1);
				//
				MsgInfo_t	*tmp_buffptr;

				long _counter=0;
				while(1){
					tmp_buffptr = m_dstBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif			
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
				if(m_dstInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
					std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
					exit(1);
				}
				// wake up other threads
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
					m_srcOpThreadLatch[m_srcOpTokenFlag[myLocalRank]]->count_down();
				else
					m_srcOpThreadAtomic[m_srcOpTokenFlag[myLocalRank]].store(0);
                m_srcOpTokenFlag[myLocalRank] = next_thread;

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			}
			else{
				// not the op thread
				int do_thread =  m_srcOpTokenFlag[myThreadRank];
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!m_srcOpThreadLatch[do_thread]->try_wait()){
						m_srcOpThreadLatch[do_thread]->wait();
					}
				}
				else{
					long _counter=0;
					while(m_srcOpThreadAtomic[do_thread].load() !=0){
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
				//
				m_srcOpTokenFlag[myLocalRank] = (m_srcOpTokenFlag[myLocalRank]+1)%m_numSrcLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif			
			}*/
		}// end multi threads	
    }// end src
    else if(myTaskid == m_dstId){
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
		if(m_numDstLocalThreads == 1){
			// only one thread in task
			MsgInfo_t *tmp_buffptr;
			long _counter=0;
			while(1){
				tmp_buffptr = m_srcBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif		
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
			if(m_srcInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
				std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
				exit(1);
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif

		}// end one thread
		else{
			// multiple threads in task
			int idx = m_dstOpTokenFlag[myLocalRank];
			int isavailable = 0;
#ifdef USE_DEBUG_ASSERT
			assert(idx<m_dstOpThreadAvailable.size());
			//assert(m_dstOpThreadFinish[idx]->load() != 0);
#endif
			if(m_dstOpThreadAvailable[idx]->compare_exchange_strong(isavailable, 1))
			{

				MsgInfo_t	*tmp_buffptr;
				long _counter=0;
				while(1){
					tmp_buffptr = m_srcBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
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
				if(m_srcInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
					std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
					exit(1);
				}


				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					m_dstOpThreadFinishLatch[idx]->count_down();
				}
				else{

					m_dstOpThreadFinish[idx]->store(0);
				}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			}
			m_dstOpTokenFlag[myLocalRank]++;
			m_dstOpTokenFlag[myLocalRank]=m_dstOpTokenFlag[myLocalRank]%m_nOps2;
			/*if(myLocalRank == m_dstOpTokenFlag[myLocalRank])
			{
				// the right thread's turn do r/w
				int next_thread = (m_dstOpTokenFlag[myLocalRank]+1) % m_numDstLocalThreads;
				m_dstOpThreadLatch[next_thread]->reset(1);
				m_dstOpThreadAtomic[next_thread].store(1);
				//
				MsgInfo_t	*tmp_buffptr;
				long _counter=0;
				while(1){
					tmp_buffptr = m_srcBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif				
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
				if(m_srcInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
					std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
					exit(1);
				}
				//
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD)
					m_dstOpThreadLatch[m_dstOpTokenFlag[myLocalRank]]->count_down();
				else
					m_dstOpThreadAtomic[m_dstOpTokenFlag[myLocalRank]].store(0);
                m_dstOpTokenFlag[myLocalRank] = next_thread;

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif

			}
			else{
				// not the op thread
				int do_thread =  m_dstOpTokenFlag[myLocalRank];
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!m_dstOpThreadLatch[do_thread]->try_wait()){
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
				 //
				 m_dstOpTokenFlag[myLocalRank] = (m_dstOpTokenFlag[myLocalRank]+1)%m_numDstLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif			
			}*/
		}// end multi threads

    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;


 }// end read()



int InprocConduit::ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag){
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
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" call Readby..."<<std::endl;
#endif
	if(myThreadRank != thread)
	{
		if(thread >= TaskManager::getCurrentTask()->getNumTotalThreads())
		{
			std::cerr<<"Error, readby thread rank "<<myThreadRank<<" out of range in task!"<<std::endl;
			exit(1);
		}
		else if(TaskManager::getCurrentTask()->isLocal(thread) == false){
			std::cerr<<"Error, readby thread rank "<<myThreadRank<<" is not on main process!"<<std::endl;
			exit(1);
		}

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"thread "<<myThreadRank<<" exit Readby!"<<std::endl;
#endif
		return 0;
	}

	if(myTaskid == m_srcId){
		MsgInfo_t *tmp_buffptr;
		long _counter=0;
		while(1){
			tmp_buffptr = m_dstBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
PRINT_TIME_NOW(*m_threadOstream)
*m_threadOstream<<"src-thread "<<myThreadRank<<" doing ReadBy msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
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
			//*m_threadOstream<<tmp_buffptr->msgTag<<"  "<<tmp_buffptr->dataPtr<<std::endl;
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
		if(m_dstInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
			exit(1);
		}
#ifdef ENABLE_OPBY_FINISH
		if(m_numSrcLocalThreads>1)
		{
			// record this op in finish set
			std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
			m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numSrcLocalThreads;
			m_readbyFinishCond.notify_all();
		}
#endif
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" finish ReadBy:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
	}
	else if(myTaskid == m_dstId){
		MsgInfo_t *tmp_buffptr;
		long _counter=0;
		while(1){
			tmp_buffptr = m_srcBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
PRINT_TIME_NOW(*m_threadOstream)
*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing ReadBy msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
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
		if(m_srcInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
			std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
			exit(1);
		}
#ifdef ENABLE_OPBY_FINISH
		if(m_numDstLocalThreads>1)
		{
			// record this op in finish set
			std::lock_guard<std::mutex> LCK4(m_readbyFinishMutex);
			m_readbyFinishSet[(tag<<LOG_MAX_TASKS)+myTaskid] = m_numDstLocalThreads;
			m_readbyFinishCond.notify_all();
		}
#endif
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" finish ReadBy:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
	}
	else{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

	return 0;

}// end writeby()


#ifdef ENABLE_OPBY_FINISH
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
#endif


int InprocConduit::ReadByFirst(void * DataPtr, DataSize_t DataSize, int tag){
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

	if(myTaskid == m_srcId){
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call ReadByFirst..."<<std::endl;
#endif
		int idx = m_srcOpFirstIdx[myLocalRank];
		int isfirst=0;
		if(m_srcOpThreadIsFirst[idx].compare_exchange_strong(isfirst, 1)){
			// the first coming thread
			// doing read op
			MsgInfo_t	*tmp_buffptr;
			long _counter=0;
			while(1){
				tmp_buffptr = m_dstBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
PRINT_TIME_NOW(*m_threadOstream)
*m_threadOstream<<"src-thread "<<myThreadRank<<" doing read msg...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
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
			if(m_dstInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
				std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
				exit(1);
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish read:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		}
		else{
			while(1){
				// last thread reset the isfirst flag to 0, for next turn't use
				int oldvalue = m_srcOpThreadIsFirst[idx].load();
				if(oldvalue == m_numSrcLocalThreads-1){
					m_srcOpThreadIsFirst[idx].store(0);
					break;
				}
				if(m_srcOpThreadIsFirst[idx].compare_exchange_strong(oldvalue, oldvalue+1))
					break;

			}
		}

		m_srcOpFirstIdx[myLocalRank]++;
		m_srcOpFirstIdx[myLocalRank]= m_srcOpFirstIdx[myLocalRank]%m_nOps;
	}
	else if(myTaskid== m_dstId){
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" call ReadByFirst..."<<std::endl;
#endif
		int idx = m_dstOpFirstIdx[myLocalRank];
		int isfirst = 0;
		if(m_dstOpThreadIsFirst[idx].compare_exchange_strong(isfirst, 1)){
			MsgInfo_t	*tmp_buffptr;
			long _counter=0;
			while(1){
				tmp_buffptr = m_srcBuffQueue->try_pop(myLocalRank);
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
#ifdef USE_DEBUG_LOG
PRINT_TIME_NOW(*m_threadOstream)
*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing read msg...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
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
			if(m_srcInnerMsgQueue->push(tmp_buffptr,myLocalRank)){
				std::cerr<<"ERROR, potential return buff timeout!"<<std::endl;
				exit(1);
			}
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
		}
		else{
			while(1){
				int oldvalue = m_dstOpThreadIsFirst[idx].load();
				if(oldvalue == m_numDstLocalThreads-1){
					m_dstOpThreadIsFirst[idx].store(0);
					break;
				}
				if(m_dstOpThreadIsFirst[idx].compare_exchange_strong(oldvalue, oldvalue+1))
					break;
			}
		}
		m_dstOpFirstIdx[myLocalRank]++;
		m_dstOpFirstIdx[myLocalRank]=m_dstOpFirstIdx[myLocalRank]%m_nOps;
	}
	else{
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}
	return 0;
}// end ReadByFirst()




}// end namespace iUtc
