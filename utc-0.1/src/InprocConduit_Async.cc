#include "InprocConduit.h"
#include "UtcBasics.h"
#include "Task_Utilities.h"
#include "TaskManager.h"

#include <cassert>
//#include <utility>
#include <thread>

namespace iUtc
{

int InprocConduit::AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag)
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
        *m_threadOstream<<"src-thread "<<myThreadRank<<" call AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
    	if(m_numSrcLocalThreads == 1)
    	{
    		// only one local thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

			AsyncWorkArgs args;
			args.DataPtr= DataPtr;
			args.DataSize= DataSize;
			args.tag = tag;
			args.WorkType = 2; // type-write
			std::unique_lock<std::mutex> LCK2(m_srcNewAsyncWorkMutex);
			m_srcAsyncWorkQueue.push_back(args);
			m_srcAsyncWriteFinishSet[tag] = std::promise<void>();
			m_srcAsyncWorkerCloseSig = false;
			if(m_srcAsyncWorkerOn == false)
			{
				//std::cout<<"async thread start"<<std::endl;
				m_srcAsyncWorkerOn = true;
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();

			}
			else
			{
				m_srcNewAsyncWork= true;
				m_srcNewAsyncWorkCond.notify_one();
			}
			LCK2.unlock();
    	}
    	else
    	{
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif

				AsyncWorkArgs args;
				args.DataPtr= DataPtr;
				args.DataSize= DataSize;
				args.tag = tag;
				args.WorkType = 2; // type-write

				std::unique_lock<std::mutex> LCK2(m_srcNewAsyncWorkMutex);
				m_srcAsyncWorkQueue.push_back(args);
				m_srcAsyncWriteFinishSet[tag] = std::promise<void>();
				m_srcAsyncWorkerCloseSig=false;
				if(m_srcAsyncWorkerOn == false)
				{
					m_srcAsyncWorkerOn=true;
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}
				else
				{
					m_srcNewAsyncWork= true;
					m_srcNewAsyncWorkCond.notify_one();
				}
				LCK2.unlock();

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
    	}// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
        return 0;

    }// end srctask
    else if(myTaskid == m_dstId)
    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call AsyncWrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1)
        {
        	// only one local thread
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
			AsyncWorkArgs args;
			args.DataPtr= DataPtr;
			args.DataSize= DataSize;
			args.tag = tag;
			args.WorkType = 2; // type-write
			std::unique_lock<std::mutex> LCK2(m_dstNewAsyncWorkMutex);
			m_dstAsyncWorkQueue.push_back(args);
			m_dstAsyncWriteFinishSet[tag] = std::promise<void>();
			m_dstAsyncWorkerCloseSig=false;
			if(m_dstAsyncWorkerOn == false)
			{
				m_dstAsyncWorkerOn=true;
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}
			else
			{
				m_dstNewAsyncWork= true;
				m_dstNewAsyncWorkCond.notify_one();
			}
			LCK2.unlock();
        }
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
		*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit AsyncWrite...:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncWrite...:("<<m_srcId<<"->"<<m_dstId<<")"<<std::endl;
#endif
				AsyncWorkArgs args;
				args.DataPtr= DataPtr;
				args.DataSize= DataSize;
				args.tag = tag;
				args.WorkType = 2; // type-write

				std::unique_lock<std::mutex> LCK2(m_dstNewAsyncWorkMutex);
				m_dstAsyncWorkQueue.push_back(args);
				m_dstAsyncWriteFinishSet[tag] = std::promise<void>();
				m_dstAsyncWorkerCloseSig=false;
				if(m_dstAsyncWorkerOn == false)
				{

					m_dstAsyncWorkerOn=true;
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}
				else
				{
					m_dstNewAsyncWork= true;
					m_dstNewAsyncWorkCond.notify_one();
				}
				LCK2.unlock();

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
			}// end first coming thread
        }// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncWrite!:("<<m_dstId<<"->"<<m_srcId<<")"<<std::endl;
#endif
        return 0;
    } // end dst task
    else
    {
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
    }//

	return 0;
}

void InprocConduit::asyncWorkerImpl(int myTaskid)
{
#ifdef USE_DEBUG_ASSERT
	assert(m_srcAsyncWorkerOn == true || m_dstAsyncWorkerOn == true);
#endif
	AsyncWorkArgs m_args;
	AsyncWorkArgs *m_args_array;
	bool first_visit = true;
	if(myTaskid == m_srcId)
	{
		std::unique_lock<std::mutex> LCK1(m_srcNewAsyncWorkMutex);
		while(1)
		{
			while(m_srcAsyncWorkQueue.size() !=0)
			{
				m_srcNewAsyncWork = false;
				int tmp_Qsize = m_srcAsyncWorkQueue.size();
				m_args_array = (AsyncWorkArgs*)malloc(sizeof(AsyncWorkArgs)*tmp_Qsize);
				for(int i=0; i<tmp_Qsize;i++)
				{
					m_args_array[i] = m_srcAsyncWorkQueue.front();
					m_srcAsyncWorkQueue.pop_front();
				}
				LCK1.unlock();

				for(int i=0;i<tmp_Qsize;i++)
				{
					m_args = m_args_array[i];
					if(m_args.WorkType == 1)
					{
						threadReadImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						m_srcAsyncReadFinishSet[m_args.tag].set_value();
					}
					else if(m_args.WorkType == 2)
					{
						//threadWriteImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						threadPWriteImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						m_srcAsyncWriteFinishSet[m_args.tag].set_value();
					}
					else
					{
						std::cerr<<"Error, undefined worktype for asyncworker!!!"<<std::endl;
						exit(1);
					}
				}
				free(m_args_array);
				LCK1.lock();
			}// end while workqueue
			m_srcNewAsyncWork = false;
			bool ret = m_srcNewAsyncWorkCond.wait_for(LCK1, std::chrono::microseconds(ASYNC_TIME_OUT),
								[=](){return m_srcNewAsyncWork == true;});
			if(ret==false || m_srcAsyncWorkerCloseSig == true)
			{
				//std::cout<<"async thread exit "<<std::endl;
				// time out or command to close
#ifdef USE_DEBUG_ASSERT
				if(m_srcAsyncWorkQueue.size())
					std::cerr<<"here"<<ret<<std::endl;
				assert(m_srcAsyncWorkQueue.size() == 0);
#endif
				m_srcNewAsyncWork =false;
				m_srcAsyncWorkerCloseSig = false;
				m_srcAsyncWorkerOn = false;
				LCK1.unlock();
				break;
			}

		}// end while(1)
	}//end src task
	else
	{
		std::unique_lock<std::mutex> LCK1(m_dstNewAsyncWorkMutex);
		while(1)
		{
			while(m_dstAsyncWorkQueue.size() !=0)
			{
				m_dstNewAsyncWork = false;
				int tmp_Qsize = m_dstAsyncWorkQueue.size();
				m_args_array = (AsyncWorkArgs*)malloc(sizeof(AsyncWorkArgs)*tmp_Qsize);
				for(int i=0; i<tmp_Qsize;i++)
				{
					m_args_array[i] = m_dstAsyncWorkQueue.front();
					m_dstAsyncWorkQueue.pop_front();
				}
				LCK1.unlock();

				for(int i=0;i<tmp_Qsize;i++)
				{
					m_args = m_args_array[i];
					if(m_args.WorkType == 1)
					{
						threadReadImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						m_dstAsyncReadFinishSet[m_args.tag].set_value();
					}
					else if(m_args.WorkType == 2)
					{
						//threadWriteImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						threadPWriteImpl(m_args.DataPtr, m_args.DataSize, m_args.tag, myTaskid);
						m_dstAsyncWriteFinishSet[m_args.tag].set_value();
					}
					else
					{
						std::cerr<<"Error, undefined worktype for asyncworker!!!"<<std::endl;
						exit(1);
					}
				}
				free(m_args_array);
				LCK1.lock();
				//

			}// end while workqueue
#ifdef USE_DEBUG_ASSERT
				assert(m_dstAsyncWorkQueue.size() == 0);
#endif
			m_dstNewAsyncWork = false;
			bool ret = m_dstNewAsyncWorkCond.wait_for(LCK1, std::chrono::microseconds(ASYNC_TIME_OUT),
								[=](){return m_dstNewAsyncWork == true;});
			if(!ret || m_dstAsyncWorkerCloseSig == true)
			{
				// time out or command to close
#ifdef USE_DEBUG_ASSERT
				if(m_dstAsyncWorkQueue.size())
					std::cerr<<"here"<<ret<<std::endl;
				assert(m_dstAsyncWorkQueue.size() == 0);
#endif
				m_dstNewAsyncWork=false;
				m_dstAsyncWorkerCloseSig = false;
				m_dstAsyncWorkerOn = false;
				LCK1.unlock();
				break;
			}
		}// end while(1)
	}//end dst task

	return;
}


int InprocConduit::threadWriteImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid)
{
	if(myTaskid == m_srcId)
	{
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
			m_srcBuffPool.insert(std::pair<MessageTag_t, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_srcAvailableBuffCount--;

			if(m_srcBuffPoolWaitlist.find(tag)!= m_srcBuffPoolWaitlist.end())
			{
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
				DataSize_t tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
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
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_srcBuffDataWrittenFlag[tmp_idx] =1;
					LCK2.unlock();
					m_srcNewBuffInsertedCond.notify_all();
				}
				else
				{
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
	}
	else
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
			m_dstBuffPool.insert(std::pair<MessageTag_t, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_dstAvailableBuffCount--;

			if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
			{
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
				DataSize_t tmp_size = (DataSize+CONDUIT_BUFFER_SIZE-1)/CONDUIT_BUFFER_SIZE;
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
					memcpy(tmp_buffinfo->dataPtr, DataPtr, DataSize);
					m_dstBuffDataWrittenFlag[tmp_idx] =1;
					LCK2.unlock();
					// notify reader that one new item inserted to buff pool
					m_dstNewBuffInsertedCond.notify_all();
				}
				else
				{
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
	}// end dsttask

	return 0;
}

int InprocConduit::threadPWriteImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid)
{
	if(myTaskid == m_srcId)
	{
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
			m_srcBuffPool.insert(std::pair<MessageTag_t, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_srcAvailableBuffCount--;

			if(m_srcBuffPoolWaitlist.find(tag)!= m_srcBuffPoolWaitlist.end())
			{
				// tag is in waitlist, means reader is already waiting for this msg.
				// passing address
				m_srcBuffPoolWaitlist.erase(tag);
			}
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
	}
	else
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
			m_dstBuffPool.insert(std::pair<MessageTag_t, BuffInfo*>(tag, tmp_buffinfo));
			// decrease availabe buff
			m_dstAvailableBuffCount--;

			if(m_dstBuffPoolWaitlist.find(tag) != m_dstBuffPoolWaitlist.end())
			{
				m_dstBuffPoolWaitlist.erase(tag);
			}
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
	}// end dsttask

	return 0;
}

void InprocConduit::AsyncWrite_Finish(int tag)
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
    if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

    if(myTaskid == m_srcId)
    {
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" wait for AsyncWrite..."<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1)
		{
			std::future<void> fut = m_srcAsyncWriteFinishSet[tag].get_future();
			fut.wait();
			m_srcNewAsyncWorkMutex.lock();
			m_srcAsyncWriteFinishSet.erase(tag);
			if(m_srcAsyncWriteFinishSet.empty() && m_srcAsyncReadFinishSet.empty())
			{
				m_srcNewAsyncWork= true;
				m_srcAsyncWorkerCloseSig=true;
				m_srcNewAsyncWorkCond.notify_one();
			}
			m_srcNewAsyncWorkMutex.unlock();
		}
		else
		{
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

				std::future<void> fut = m_srcAsyncWriteFinishSet[tag].get_future();
				fut.wait();
				m_srcNewAsyncWorkMutex.lock();
				m_srcAsyncWriteFinishSet.erase(tag);
				if(m_srcAsyncWriteFinishSet.empty() && m_srcAsyncReadFinishSet.empty())
				{
					m_srcNewAsyncWork= true;
					m_srcAsyncWorkerCloseSig=true;
					m_srcNewAsyncWorkCond.notify_one();
				}
				m_srcNewAsyncWorkMutex.unlock();

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
		}// end several threads

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif

    }// end srctask
    else
    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" wait for AsyncWrite..."<<std::endl;
#endif
        if(m_numDstLocalThreads == 1)
        {
        	// only one local thread
			std::future<void> fut = m_dstAsyncWriteFinishSet[tag].get_future();
			fut.wait();
			m_dstNewAsyncWorkMutex.lock();
			m_dstAsyncWriteFinishSet.erase(tag);
			if(m_dstAsyncWriteFinishSet.empty() && m_dstAsyncReadFinishSet.empty())
			{
				m_dstNewAsyncWork = true;
				m_dstAsyncWorkerCloseSig = true;
				m_dstNewAsyncWorkCond.notify_one();
			}
			m_dstNewAsyncWorkMutex.unlock();
        }
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


				std::future<void> fut = m_dstAsyncWriteFinishSet[tag].get_future();
				fut.wait();
				m_dstNewAsyncWorkMutex.lock();
				m_dstAsyncWriteFinishSet.erase(tag);
				if(m_dstAsyncWriteFinishSet.empty() && m_dstAsyncReadFinishSet.empty())
				{
					m_dstNewAsyncWork = true;
					m_dstAsyncWorkerCloseSig = true;
					m_dstNewAsyncWorkCond.notify_one();
				}
				m_dstNewAsyncWorkMutex.unlock();


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
			}// end first coming thread
        }// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish wait AsyncWrite!"<<std::endl;
#endif
    }// end dsttask

	return;
}


int InprocConduit::AsyncRead(void* DataPtr, DataSize_t DataSize, int tag)
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" call AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1)
		{
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
			AsyncWorkArgs args;
			args.DataPtr= DataPtr;
			args.DataSize= DataSize;
			args.tag = tag;
			args.WorkType = 1; // type-read
			std::unique_lock<std::mutex> LCK2(m_srcNewAsyncWorkMutex);
			m_srcAsyncWorkQueue.push_back(args);
			m_srcAsyncReadFinishSet[tag] = std::promise<void>();
			m_srcAsyncWorkerCloseSig=false;
			if(m_srcAsyncWorkerOn == false)
			{
				m_srcAsyncWorkerOn = true;
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}
			else
			{
				m_srcNewAsyncWork= true;
				m_srcNewAsyncWorkCond.notify_one();
			}
			LCK2.unlock();
		}
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
		*m_threadOstream<<"src-thread "<<myThreadRank<<" exit AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
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

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_threadOstream)
    *m_threadOstream<<"src-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif

    			AsyncWorkArgs args;
    			args.DataPtr= DataPtr;
    			args.DataSize= DataSize;
    			args.tag = tag;
    			args.WorkType = 1; // type-read
    			std::unique_lock<std::mutex> LCK2(m_srcNewAsyncWorkMutex);
    			m_srcAsyncWorkQueue.push_back(args);
    			m_srcAsyncReadFinishSet[tag] = std::promise<void>();
    			m_srcAsyncWorkerCloseSig=false;
    			if(m_srcAsyncWorkerOn == false)
    			{
    				m_srcAsyncWorkerOn = true;
    				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
    			}
    			else
    			{
    				m_srcNewAsyncWork= true;
    				m_srcNewAsyncWorkCond.notify_one();
    			}
    			LCK2.unlock();


#ifdef USE_DEBUG_ASSERT
                assert(m_srcOpRotateFinishFlag[counteridx] ==0);
#endif
                m_srcOpRotateFinishFlag[counteridx]++;
                m_srcOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
                m_srcOpFinishCond.notify_all();

				LCK1.unlock();
			}// end first coming thread

		}// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"src-thread "<<myThreadRank<<" finish AsyncRead:("<<m_srcId<<"<-"<<m_dstId<<")"<<std::endl;
#endif
        return 0;

	}// end src task
    else if(myTaskid == m_dstId)
    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" call AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        if(m_numDstLocalThreads == 1)
        {
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
			AsyncWorkArgs args;
			args.DataPtr= DataPtr;
			args.DataSize= DataSize;
			args.tag = tag;
			args.WorkType = 1; // type-read
			std::unique_lock<std::mutex> LCK2(m_dstNewAsyncWorkMutex);
			m_dstAsyncWorkQueue.push_back(args);
			m_dstAsyncReadFinishSet[tag] = std::promise<void>();
			m_dstAsyncWorkerCloseSig=false;
			if(m_dstAsyncWorkerOn == false)
			{
				m_dstAsyncWorkerOn = true;
				std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
			}
			else
			{
				m_dstNewAsyncWork= true;
				m_dstNewAsyncWorkCond.notify_one();
			}
			LCK2.unlock();
        }
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
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" exit AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
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
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"dst-thread "<<myThreadRank<<" doing AsyncRead...:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
				AsyncWorkArgs args;
				args.DataPtr= DataPtr;
				args.DataSize= DataSize;
				args.tag = tag;
				args.WorkType = 1; // type-read
				std::unique_lock<std::mutex> LCK2(m_dstNewAsyncWorkMutex);
				m_dstAsyncWorkQueue.push_back(args);
				m_dstAsyncReadFinishSet[tag] = std::promise<void>();
				m_dstAsyncWorkerCloseSig=false;
				if(m_dstAsyncWorkerOn == false)
				{
					m_dstAsyncWorkerOn = true;
					std::thread(&InprocConduit::asyncWorkerImpl, this, myTaskid).detach();
				}
				else
				{
					m_dstNewAsyncWork= true;
					m_dstNewAsyncWorkCond.notify_one();
				}
				LCK2.unlock();

#ifdef USE_DEBUG_ASSERT
                assert(m_dstOpRotateFinishFlag[counteridx] ==0);
#endif
                m_dstOpRotateFinishFlag[counteridx]++;
                m_dstOpRotateCounterIdx[myThreadRank] = (counteridx +1)%(m_noFinishedOpCapacity+1);
                m_dstOpFinishCond.notify_all();
				LCK1.unlock();
			}
        }// end several threads

#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish AsyncRead:("<<m_dstId<<"<-"<<m_srcId<<")"<<std::endl;
#endif
        return 0;
    }// end dst task
    else
    {
    	std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
    }
	return 0;
}

int InprocConduit::threadReadImpl(void* DataPtr, DataSize_t DataSize, int tag, int myTaskid)
{
	if(myTaskid == m_srcId)
	{
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
					std::cerr<<"Error, async reader wait time out!"<<"src-task"<<myTaskid<<std::endl;
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
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
				// notify writer that read finish
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
				LCK3.unlock();
			}
		}// end for tag not in the pool
		else
		{
			// find tag, means writer already comes

			tmp_buffinfo = m_dstBuffPool[tag];
#ifdef USE_DEBUG_ASSERT
			assert(tmp_buffinfo->callingReadThreadCount == 0);
			assert(tmp_buffinfo->callingWriteThreadCount >0);
#endif
			tmp_buffinfo->callingReadThreadCount =1;
			LCK2.unlock();

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
				std::unique_lock<std::mutex> LCK3(m_dstBuffAccessMutex[tmp_buffinfo->buffIdx]);
				m_dstBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
				// notify writer that read finish
				m_dstBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
				LCK3.unlock();
			}

		}//end for tag in the pool
	}// end src task
	else
	{
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
				std::cerr<<"Error, async reader wait time out!"<<"dst-task "<<myTaskid<<std::endl;
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
			std::unique_lock<std::mutex> LCK3(m_srcBuffAccessMutex[tmp_buffinfo->buffIdx]);
			m_srcBuffDataReadFlag[tmp_buffinfo->buffIdx] =1;
			// notify writer that read finish
			m_srcBuffDataReadCond[tmp_buffinfo->buffIdx].notify_one();
			LCK3.unlock();
		}
	}// end dst task

	return 0;
}

void InprocConduit::AsyncRead_Finish(int tag)
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
    if(myTaskid != m_srcId && myTaskid!=m_dstId)
	{
		std::cerr<<"Error, conduit doesn't associate to calling task!"<<std::endl;
		exit(1);
	}

    if(myTaskid == m_srcId)
    {
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" wait for AsyncRead..."<<std::endl;
#endif
		if(m_numSrcLocalThreads == 1)
		{
			std::future<void> fut = m_srcAsyncReadFinishSet[tag].get_future();
			fut.wait();
			m_srcNewAsyncWorkMutex.lock();
			m_srcAsyncReadFinishSet.erase(tag);
			if(m_srcAsyncWriteFinishSet.empty() && m_srcAsyncReadFinishSet.empty())
			{
				m_srcNewAsyncWork= true;
				m_srcAsyncWorkerCloseSig=true;
				m_srcNewAsyncWorkCond.notify_one();
			}
			m_srcNewAsyncWorkMutex.unlock();
		}
		else
		{
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

				std::future<void> fut = m_srcAsyncReadFinishSet[tag].get_future();
				fut.wait();
				m_srcNewAsyncWorkMutex.lock();
				m_srcAsyncReadFinishSet.erase(tag);
				if(m_srcAsyncWriteFinishSet.empty() && m_srcAsyncReadFinishSet.empty())
				{
					m_srcNewAsyncWork= true;
					m_srcAsyncWorkerCloseSig=true;
					m_srcNewAsyncWorkCond.notify_one();
				}
				m_srcNewAsyncWorkMutex.unlock();


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
		}// end several threads

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	*m_threadOstream<<"src-thread "<<myThreadRank<<" finish wait AsyncRead!"<<std::endl;
#endif

    }// end srctask
    else
    {
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" wait for AsyncRead..."<<std::endl;
#endif
        if(m_numDstLocalThreads == 1)
        {
        	// only one local thread
        	std::future<void> fut = m_dstAsyncReadFinishSet[tag].get_future();
			fut.wait();
			m_dstNewAsyncWorkMutex.lock();
			m_dstAsyncReadFinishSet.erase(tag);
			if(m_dstAsyncWriteFinishSet.empty() && m_dstAsyncReadFinishSet.empty())
			{
				m_dstNewAsyncWork= true;
				m_dstAsyncWorkerCloseSig=true;
				m_dstNewAsyncWorkCond.notify_one();
			}
			m_dstNewAsyncWorkMutex.unlock();
        }
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


				std::future<void> fut = m_dstAsyncReadFinishSet[tag].get_future();
				fut.wait();
				m_dstNewAsyncWorkMutex.lock();
				m_dstAsyncReadFinishSet.erase(tag);
				if(m_dstAsyncWriteFinishSet.empty() && m_dstAsyncReadFinishSet.empty())
				{
					m_dstNewAsyncWork= true;
					m_dstAsyncWorkerCloseSig=true;
					m_dstNewAsyncWorkCond.notify_one();
				}
				m_dstNewAsyncWorkMutex.unlock();

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
			}// end first coming thread
        }// end several threads
#ifdef USE_DEBUG_LOG
        PRINT_TIME_NOW(*m_threadOstream)
        *m_threadOstream<<"dst-thread "<<myThreadRank<<" finish wait AsyncRead!"<<std::endl;
#endif
    }// end dsttask

	return;
}


}// end namespce iUtc
