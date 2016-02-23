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

			// fetch one msg from buffer queue
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
			// multiple threads in task
			if(myThreadRank == m_srcOpTokenFlag[myThreadRank])
			{
				// the right thread's turn do r/w
				int next_thread = (m_srcOpTokenFlag[myThreadRank]+1) % m_numSrcLocalThreads;
				m_srcOpThreadLatch[next_thread]->reset(1);
				m_srcOpThreadAtomic[next_thread].store(1);
				//
				MsgInfo_t	*tmp_buffptr;
				// fetch one msg from buffer queue
				tmp_buffptr = m_dstBuffQueue->pop(myLocalRank);
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
					m_srcOpThreadLatch[m_srcOpTokenFlag[myThreadRank]]->count_down();
				else
					m_srcOpThreadAtomic[m_srcOpTokenFlag[myThreadRank]].store(0);
                m_srcOpTokenFlag[myThreadRank] = next_thread;

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
					while(m_srcOpThreadAtomic[do_thread].load() !=0){
						_mm_pause();
					}
				}
				//
				m_srcOpTokenFlag[myThreadRank] = (m_srcOpTokenFlag[myThreadRank]+1)%m_numSrcLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" exit read...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif			
			}
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

			// fetch one msg from buffer queue
			tmp_buffptr = m_srcBuffQueue->pop(myLocalRank);
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
					while((tmp_buffptr = m_srcBuffQueue->pop(myLocalRank))!=nullptr){
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
			if(myThreadRank == m_dstOpTokenFlag[myThreadRank])
			{
				// the right thread's turn do r/w
				int next_thread = (m_dstOpTokenFlag[myThreadRank]+1) % m_numDstLocalThreads;
				m_dstOpThreadLatch[next_thread]->reset(1);
				m_dstOpThreadAtomic[next_thread].store(1);
				//
				MsgInfo_t	*tmp_buffptr;
				// fetch one msg from buffer queue
				tmp_buffptr = m_srcBuffQueue->pop(myLocalRank);
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
						while((tmp_buffptr = m_dstBuffQueue->pop(myLocalRank))!=nullptr){
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
					m_dstOpThreadLatch[m_dstOpTokenFlag[myThreadRank]]->count_down();
				else
					m_dstOpThreadAtomic[m_dstOpTokenFlag[myThreadRank]].store(0);
                m_dstOpTokenFlag[myThreadRank] = next_thread;

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish read:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif

			}
			else{
				// not the op thread
				int do_thread =  m_dstOpTokenFlag[myThreadRank];
				if(DataSize > CONDUIT_LATCH_ATOMI_THRESHHOLD){
					if(!m_dstOpThreadLatch[do_thread]->try_wait()){
						m_dstOpThreadLatch[do_thread]->wait();
					}
				}
				else{
					while(m_dstOpThreadAtomic[do_thread].load() !=0){
						_mm_pause();
					}
				}
				 //
				 m_dstOpTokenFlag[myThreadRank] = (m_dstOpTokenFlag[myThreadRank]+1)%m_numDstLocalThreads;
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" exit read...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif			
			}
		}// end multi threads

    }// end dst
    else{
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
        exit(1);
    }

    return 0;


 }// end read()


}// end namespace iUtc
