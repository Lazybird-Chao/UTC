/*
 * TaskUtilities.cc
 *
 *  Created on: Jun 15, 2016
 *      Author: chaoliu
 */
#include "TaskUtilities.h"
#include "TaskManager.h"
#include "RootTask.h"
#include "UniqueExeTag.h"

#include <functional>
#include <atomic>
#include "boost/filesystem.hpp"

namespace iUtc{

/* Task utility methods */

std::ofstream* getProcOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid != 0)
	{
		// this function must be called in main thread, not in some task's thread
		// main thread in a process is actually root, root task id is 0
		std::cerr<<"Error, getProcOstream() must be called in main thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	RootTask *root = TaskManager::getRootTask();
	std::ofstream *procOstream = root->getProcOstream();
	if(!procOstream)
	{
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/Proc";
		filename.append(std::to_string(root->getCurrentProcRank()));
		filename.append(".log");
		procOstream = new std::ofstream(filename);
		root->setProcOstream(*procOstream);

	}
	//return std::ref(*procOstream);
	return procOstream;
}

std::ofstream* getThreadOstream()
{
	int currentTaskid = TaskManager::getCurrentTaskId();
	if(currentTaskid == 0)
	{
		std::cerr<<"Error, getThreadOstream() must be called in task's thread\n";
		//std::ofstream output;
		//return std::ref(output);
		return nullptr;
	}
	ThreadPrivateData *tpd = TaskBase::getThreadPrivateData();
	if(!tpd->threadOstream)
	{
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/";
		filename.append((TaskManager::getCurrentTask())->getName());
		filename.append("-");
		filename.append(std::to_string(TaskManager::getCurrentTaskId()));
		filename.append("-thread");
		filename.append(std::to_string(TaskManager::getCurrentThreadRankinTask()));
		filename.append(".log");
		tpd->threadOstream = new std::ofstream(filename);
	}

	//return std::ref(*(tpd->threadOstream));
	return tpd->threadOstream;

}

int getTaskId()
{
	static thread_local int taskid = -1;
	if(taskid == -1)
	{
		taskid = TaskManager::getCurrentTaskId();
	}
	return taskid;
}

int getParentTaskId()
{
	static thread_local int parentTid = -1;
	if(parentTid == -1)
	{
		parentTid = TaskManager::getParentTaskId();
	}
	return parentTid;
}

int getGrank()
{
	static thread_local int threadrank=-1;
	if(threadrank ==-1)
	{
		threadrank = TaskManager::getCurrentThreadRankinTask();
	}
	return threadrank;
}

int getPrank()
{
	static thread_local int procrank = -1;
	if(procrank == -1)
	{
		procrank = TaskManager::getCurrentProcessRankinTask();
	}
	return procrank;
}

int getLrank()
{
	static thread_local int localrank = -1;
	if(localrank == -1)
	{
		localrank = TaskManager::getCurrentThreadRankInLocal();
	}
	return localrank;
}


int getLsize()
{
	static thread_local int localthreads = -1;
	if(localthreads ==-1)
	{
		localthreads = TaskManager::getCurrentTask()->getNumLocalThreads();
	}
	return localthreads;
}

int getGsize()
{
	static thread_local int globalthreads = -1;
	if(globalthreads == -1)
	{
		globalthreads = TaskManager::getCurrentTask()->getNumTotalThreads();
	}
	return globalthreads;
}

int getPsize(){
	static thread_local int numProcs = -1;
	if(numProcs ==-1){
		numProcs = TaskManager::getCurrentTask()->getNumProcesses();
	}
	return numProcs;
}


TaskBase* getCurrentTask()
{
	static thread_local TaskBase* currentTaskPtr= nullptr;
	if(!currentTaskPtr)
	{
		currentTaskPtr = TaskManager::getCurrentTask();
	}
	return currentTaskPtr;
}

TaskBase* getParentTask()
{
	static thread_local TaskBase* ParentTaskPtr= nullptr;
	if(!ParentTaskPtr)
	{
		ParentTaskPtr = TaskManager::getParentTask();
	}
	return ParentTaskPtr;
}



bool getUniqueExecution(){
	static thread_local UniqueExeTag *uniqueExeTagPtr = nullptr;
	static thread_local int mylocalrank = -1;
	if(mylocalrank== -1){

		mylocalrank = TaskManager::getCurrentThreadRankInLocal();
		uniqueExeTagPtr = TaskBase::getThreadPrivateData()->taskUniqueExeTagObj;
	}

	return uniqueExeTagPtr->getUniqueExe(mylocalrank);
}


/*
void BcastInTask(void* Data, DataSize_t DataSize, Rank_t rootthread, Rank_t rootproc){
	static thread_local int currentProcRank = -1;
	static thread_local int currentThreadRank = -1;
	static thread_local int currentThreadLocalRank = -1;
	static thread_local ThreadPrivateData* tpd;
	static thread_local SpinBarrier* sbarrier;
	static thread_local int numProcs = -1;
	static thread_local int numLocalThreads=-1;
	if(currentProcRank == -1){
		currentProcRank= getPrank();
		currentThreadRank = getTrank();
		currentThreadLocalRank = getLrank();
		tpd = TaskBase::getThreadPrivateData();
		sbarrier = TaskManager::getTaskInfo()->spinBarrierObjPtr;
		numProcs = getPsize();
		numLocalThreads = getLsize();
	}
	void **bcastDataBufferPtr = tpd->bcastDataBuffer;
	std::atomic<int>* bcastDataReadyPtr = tpd->bcastDataReady;
	if(currentThreadRank == rootthread){
		// the bcast thread
		MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
		if(numLocalThreads > 1){
			while(bcastDataReadyPtr->load()!=0){
				_mm_pause();
			}
			assert(*bcastDataBufferPtr == nullptr);
			*bcastDataBufferPtr = Data;
			bcastDataReadyPtr->store(1);
		}
		if(numProcs>1){
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Bcast(Data, (int)DataSize, datatype, rootproc, taskcomm);
#endif
		}
		if(numLocalThreads > 1){
			sbarrier->wait();
		}

	}
	else if(currentProcRank==rootproc){
		// not bcaster, but in same process with bcaster
		while(bcastDataReadyPtr->load()==0){
			_mm_pause();
		}
		if(Data != *bcastDataBufferPtr){
			memcpy(Data, *bcastDataBufferPtr, DataSize);
		}
		bcastDataReadyPtr->fetch_add(1);
		int nthreads = numLocalThreads;
		if(bcastDataReadyPtr->compare_exchange_strong(nthreads, 0)){
			*bcastDataBufferPtr = nullptr;
		}
		sbarrier->wait();
	}
	else{
		// other thread on other process
		std::atomic<int>* bcastAvailable = tpd->bcastAvailable;
		int isavailable =0;
		if(bcastAvailable->compare_exchange_strong(isavailable, 1)){
			// first threads
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Bcast(Data, (int)DataSize, datatype, rootproc, taskcomm);
#endif
			if(numLocalThreads > 1){
				while(bcastDataReadyPtr->load()!=0){
					_mm_pause();
				}
				assert(*bcastDataBufferPtr == nullptr);
				*bcastDataBufferPtr = Data;
				bcastDataReadyPtr->store(1);

				sbarrier->wait();
			}
			else{
				bcastAvailable->store(0);
			}
		}
		else{
			// other late threads
			while(bcastDataReadyPtr->load()==0){
				_mm_pause();
			}
			if(Data != *bcastDataBufferPtr){
				memcpy(Data, *bcastDataBufferPtr, DataSize);
			}
			bcastDataReadyPtr->fetch_add(1);
			int nthreads = numLocalThreads;
			if(bcastDataReadyPtr->compare_exchange_strong(nthreads, 0)){
				bcastAvailable->store(0);
				*bcastDataBufferPtr = nullptr;
			}
			sbarrier->wait();
		}

	}

}// end bcastintask()
*/

void SharedDataBcast(void* Data, DataSize_t DataSize, Rank_t rootthread){
	static thread_local int currentProcRank = -1;
	static thread_local int currentThreadRank = -1;
	static thread_local SpinBarrier* sbarrier;
	static thread_local int numProcs = -1;
	static thread_local int numLocalThreads=-1;
	int rootproc;
	if(currentProcRank == -1){
		currentProcRank= getPrank();
		currentThreadRank = getGrank();
		sbarrier = TaskManager::getTaskInfo()->spinBarrierObjPtr;
		numProcs = getPsize();
		numLocalThreads = getLsize();
	}
	rootproc = TaskManager::getCurrentTask()->getProcRankOfThread(rootthread);

	if(currentThreadRank == rootthread){
		if(numProcs>1){
			MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Bcast(Data, (int)DataSize, datatype, rootproc, *taskcomm);
#endif
		}
		sbarrier->wait();
	}
	else if(currentProcRank == rootproc){
		sbarrier->wait();
	}
	else{
		if(numLocalThreads>1){
			std::atomic<int>* bcastAvailable = TaskBase::getThreadPrivateData()->bcastAvailable;
			int isavailable = 0;
			if(bcastAvailable->compare_exchange_strong(isavailable, 1)){
				MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
				// TODO: solving int(datasize) problem
				MPI_Datatype datatype=MPI_CHAR;
				if(DataSize > ((unsigned)1<<31)-1){
					DataSize = (DataSize+3)/4;
					datatype = MPI_INT;
				}
				MPI_Bcast(Data, (int)DataSize, datatype, rootproc, *taskcomm);
#endif
				sbarrier->wait();
			}
			else{
				bcastAvailable->fetch_add(1);
				int nthreads = numLocalThreads;
				bcastAvailable->compare_exchange_strong(nthreads, 0);
				sbarrier->wait();
			}
		}
		else{
			MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Bcast(Data, (int)DataSize, datatype, rootproc, *taskcomm);
#endif
		}
	}
	return;

}// end ShareadDataBcast()

void SharedDataGather(void *DataSend, DataSize_t DataSize, void *DataGathered,
		Rank_t rootthread){
	static thread_local int currentProcRank = -1;
	static thread_local int currentThreadRank = -1;
	static thread_local SpinBarrier* sbarrier=nullptr;
	//static thread_local Barrier* barrier = nullptr;
	static thread_local int numProcs = -1;
	static thread_local int numLocalThreads=-1;
	int rootproc;
	if(currentProcRank == -1){
		currentProcRank= getPrank();
		currentThreadRank = getGrank();
		sbarrier = TaskManager::getTaskInfo()->spinBarrierObjPtr;
		//barrier = TaskManager::getTaskInfo()->barrierObjPtr;
		numProcs = getPsize();
		numLocalThreads = getLsize();
	}
	rootproc = TaskManager::getCurrentTask()->getProcRankOfThread(rootthread);

	if(currentThreadRank == rootthread){
		if(numProcs >1){
			MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Gather(DataSend, DataSize, datatype, DataGathered, DataSize, datatype,
					rootproc, *taskcomm);
#endif
		}
		else{
			memcpy(DataGathered, DataSend, DataSize);
		}
		sbarrier->wait();
		//barrier->synch_intra(0);
	}
	else if(currentProcRank == rootproc){
		sbarrier->wait();
		//barrier->synch_intra(0);
	}
	else{
		if(numLocalThreads>1){
			std::atomic<int>* gatherAvailable = TaskBase::getThreadPrivateData()->gatherAvailable;
			int isavailable=0;
			if(gatherAvailable->compare_exchange_strong(isavailable, 1)){
				MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
				// TODO: solving int(datasize) problem
				MPI_Datatype datatype=MPI_CHAR;
				if(DataSize > ((unsigned)1<<31)-1){
					DataSize = (DataSize+3)/4;
					datatype = MPI_INT;
				}
				MPI_Gather(DataSend, DataSize, datatype, DataGathered, DataSize, datatype,
						rootproc, *taskcomm);
#endif
				sbarrier->wait();
				//barrier->synch_intra(0);
			}
			else{
				gatherAvailable->fetch_add(1);
				int nthreads = numLocalThreads;
				gatherAvailable->compare_exchange_strong(nthreads, 0);
				sbarrier->wait();
				//barrier->synch_intra(0);
			}
		}
		else{
			MPI_Comm *taskcomm = TaskManager::getTaskInfo()->commPtr;
#ifdef USE_MPI_BASE
			// TODO: solving int(datasize) problem
			MPI_Datatype datatype=MPI_CHAR;
			if(DataSize > ((unsigned)1<<31)-1){
				DataSize = (DataSize+3)/4;
				datatype = MPI_INT;
			}
			MPI_Gather(DataSend, DataSize, datatype, DataGathered, DataSize, datatype,
					rootproc, *taskcomm);
#endif
		}
	}
	return;
}// end ShareadDataGather()


}// end namespace iUtc

