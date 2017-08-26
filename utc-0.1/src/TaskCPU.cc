#include "TaskCPU.h"
#include "UtcContext.h"
#include "TaskManager.h"

namespace iUtc{

TaskCPU::TaskCPU(TaskType taskType,
				int numLocalThreads,
				 int currentProcessRank,
				 int numProcesses,
				 int numTotalThreads,
				 std::vector<ThreadRank_t> tRankList,
				 std::vector<ThreadId_t> *LocalThreadList,
				 std::map<ThreadId_t, ThreadRank_t> *LocalThreadRegistry,
				 std::map<ThreadRank_t, int> *ThreadRank2Local,
				 std::ofstream *procOstream,
				 TaskInfo *commonThreadInfo,
				 ThreadPrivateData *commonThreadPrivateData,
				 boost::thread_specific_ptr<ThreadPrivateData>* threadPrivateData,
				 UserTaskBase* realUserTaskObj){

	m_taskType = taskType;
	m_numLocalThreads = numLocalThreads;
	m_currentProcessRank = currentProcessRank;
	m_numProcesses = numProcesses;
	m_numTotalThreads = numTotalThreads;
	m_tRankList = tRankList;
	m_LocalThreadList = LocalThreadList;
	m_LocalThreadRegistry = LocalThreadRegistry;
	m_ThreadRank2Local = ThreadRank2Local;
	m_procOstream = procOstream;
	m_commonTaskInfo = commonThreadInfo;
	m_commonThreadPrivateData = commonThreadPrivateData;
	m_threadPrivateData = threadPrivateData;
	m_realUserTaskObj = realUserTaskObj;

	m_taskThreads.clear();

	m_threadSync = nullptr;
	m_jobDoneWait = nullptr;
	m_jobQueue.clear();
	m_jobHandleQueue.clear();
	m_threadJobIdx = nullptr;

	m_activeLocalThreadCount= m_numLocalThreads;

}

TaskCPU::~TaskCPU(){
	for(auto& th : m_taskThreads)
	{
		if(th.joinable())
		{
			ThreadId_t id = th.get_id();
			th.join();
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_procOstream)
		(*m_procOstream)<<"cputask thread "<<id<<" join on "<<getpid()
				<<" for ["<<m_Name<<"]"<<std::endl;
#endif
		}
	}
	m_taskThreads.clear();


	m_numLocalThreads=0;
	m_currentProcessRank = -1;
	m_numProcesses = 0;
	m_tRankList.clear();
	m_LocalThreadList= nullptr;
	m_LocalThreadRegistry= nullptr;
	m_ThreadRank2Local = nullptr;
	m_procOstream = nullptr;

	if(m_threadSync)
		delete m_threadSync;
	if(m_jobDoneWait)
		delete m_jobDoneWait;
	m_jobQueue.clear();
	m_jobHandleQueue.clear();
	if(m_threadJobIdx)
		delete m_threadJobIdx;

}

void TaskCPU::launchThreads(){
	m_threadSync = new boost::barrier(m_numLocalThreads);
	m_jobDoneWait = new boost::latch(m_numLocalThreads);
	m_jobQueue.clear();
	m_jobHandleQueue.clear();
	m_threadJobIdx = new int[m_numLocalThreads];
	for(int i=0;i<m_numLocalThreads;i++)
		m_threadJobIdx[i]=0;
	//std::cout<<m_numLocalThreads<<" "<<ERROR_LINE<<std::endl;
	////
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"start launching cputask threads for ["<<m_Name<<"] on proc "<<m_processRank
			<<"..."<<std::endl;
#endif
	//
	/*std::unique_lock<std::mutex> LCK(m_activeLocalThreadMutex);
	m_activeLocalThreadCount = m_numLocalThreads;
	LCK.unlock();*/
	int trank=0;
	ThreadId_t id;
	for(int i=0; i<m_numLocalThreads; i++)
	{
		trank=m_tRankList[i];
#ifdef USE_DEBUG_LOG
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/";
		filename.append(m_Name);
		filename.append("-");
		filename.append(std::to_string(m_TaskId));
		filename.append("-thread");
		filename.append(std::to_string(trank));
		filename.append(".log");
		std::ofstream *output = new std::ofstream(filename);
		filename.clear();
#else
		std::ofstream *output = nullptr;
#endif

		m_ThreadRank2Local->insert(std::pair<ThreadRank_t, int>(trank, i));
		// crteate a thread
#ifdef SET_CPU_AFFINITY
		int hardcoreId = UtcContext::HARDCORES_ID_FOR_USING;
		UtcContext::HARDCORES_ID_FOR_USING = (UtcContext::HARDCORES_ID_FOR_USING+1)%UtcContext::HARDCORES_TOTAL_CURRENT_NODE;
		m_taskThreads.push_back(std::thread(&TaskCPU::threadImpl, this, trank, i, output, hardcoreId));
#else
		m_taskThreads.push_back(std::thread(&TaskCPU::threadImpl, this, trank, i, output, -1));
#endif
		//m_taskThreads[i].detach();
		// record some info
		id = m_taskThreads.back().get_id();
		m_LocalThreadList->push_back(id);
		m_LocalThreadRegistry->insert(std::pair<ThreadId_t, ThreadRank_t>(id, trank));
		//std::cout<<trank<<" "<<ERROR_LINE<<std::endl;
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<m_numLocalThreads<<" cputask threads for ["<<m_Name<<"] launched on proc "<<m_processRank
			<<std::endl;
#endif

	return;
}

void TaskCPU::threadImpl(ThreadRank_t trank,
						ThreadRank_t lrank,
						std::ofstream *output,
						int hardcoreId){
	//
	std::ofstream *m_threadOstream = output;
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	(*m_threadOstream)<<"thread("<<trank<<") "<<std::this_thread::get_id()<<" start up on "
			<<getpid()<<"..."<<std::endl;
#endif

#ifdef SET_CPU_AFFINITY
	assert(hardcoreId>=0);
	std::vector<int> cpus;
	cpus.push_back(hardcoreId);
	setAffinity(cpus);
#endif

//#define SET_CPU_BIND_TO_ALL
#ifdef SET_CPU_BIND_TO_ALL
	std::cout<<"hardcore total: "<<UtcContext::HARDCORES_TOTAL_CURRENT_NODE<<std::endl;
	std::vector<int> cpus;
	for(int i=0; i<UtcContext::HARDCORES_TOTAL_CURRENT_NODE; i++)
		cpus.push_back(i);
	setAffinity(cpus);
#endif

	// create task info in this thread TSS
	TaskInfo* taskInfoPtr = new TaskInfo();
	taskInfoPtr->pRank = m_commonTaskInfo->pRank;
	taskInfoPtr->parentTaskId = m_commonTaskInfo->parentTaskId;
	taskInfoPtr->tRank = trank;
	taskInfoPtr->lRank = lrank;
	taskInfoPtr->taskId = m_commonTaskInfo->taskId;
	taskInfoPtr->threadId = std::this_thread::get_id();
	taskInfoPtr->barrierObjPtr = m_commonTaskInfo->barrierObjPtr;
	taskInfoPtr->spinBarrierObjPtr = m_commonTaskInfo->spinBarrierObjPtr;
	taskInfoPtr->fastBarrierObjPtr = m_commonTaskInfo->fastBarrierObjPtr;
	taskInfoPtr->commPtr = m_commonTaskInfo->commPtr;
	taskInfoPtr->mpigroupPtr = m_commonTaskInfo->mpigroupPtr;
	taskInfoPtr->worldCommPtr = m_commonTaskInfo->worldCommPtr;
	taskInfoPtr->worldGroupPtr = m_commonTaskInfo->worldGroupPtr;
	taskInfoPtr->worldRankToGrouprRank = m_commonTaskInfo->worldRankToGrouprRank;
	TaskManager::setTaskInfo(taskInfoPtr);
	//std::cout<<trank<<" "<<ERROR_LINE<<std::endl;
	// create ThreadPrivateData structure
	ThreadPrivateData *threadPrivateData = new ThreadPrivateData();
	threadPrivateData->threadOstream = output;
	threadPrivateData->taskUniqueExeTagObj = m_commonThreadPrivateData->taskUniqueExeTagObj;
	threadPrivateData->bcastAvailable = m_commonThreadPrivateData->bcastAvailable;
	threadPrivateData->gatherAvailable = m_commonThreadPrivateData->gatherAvailable;
	m_threadPrivateData->reset(threadPrivateData);
	//std::cout<<trank<<" "<<ERROR_LINE<<std::endl;
	// do preInit()
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	(*m_threadOstream)<<"thread("<<trank<<") "<<std::this_thread::get_id()<<" doing preInit..."<<std::endl;
#endif
	m_threadSync->count_down_and_wait();
	m_realUserTaskObj->preInit(lrank,
							trank,
							m_commonTaskInfo->pRank,
							m_numLocalThreads,
							m_numProcesses,
							m_numTotalThreads,
							m_commonTaskInfo->worldRankToGrouprRank, nullptr);

	while(1){
		std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
		m_jobExecCond.wait(LCK1,
				[=](){return m_jobQueue.size()> m_threadJobIdx[lrank];});
		int tmpJobHandle = m_jobQueue[m_threadJobIdx[lrank]];
		LCK1.unlock();

		switch(tmpJobHandle){
		case threadJobType::job_init:
			m_InitHandle();
			threadSync(lrank);
			break;
		case threadJobType::job_run:
			m_RunHandle();
			//threadSync();
			break;
		case threadJobType::job_finish:
			break;
		case threadJobType::job_wait:
			threadWait();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	(*m_threadOstream)<<"thread("<<trank<<") "<<"called wait!"<<std::endl;
#endif
			break;
		case threadJobType::job_user_defined:
			// run user specified function
			//m_ExecHandle();
			m_jobHandleQueue[m_threadJobIdx[lrank]]();
			//threadSync();
			break;
		default:
			std::cerr<<"Error! Undefined Task job type!"<<std::endl;
			break;
		}
		m_threadJobIdx[lrank]++;
		if(tmpJobHandle == threadJobType::job_finish)
			break;
	}

	// do preExit()
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	(*m_threadOstream)<<"thread("<<trank<<") "<<std::this_thread::get_id()<<" doing preExit..."<<std::endl;
#endif
	m_realUserTaskObj->preExit();

	// do thread exit clear work at this point
	threadExit(trank);
	return;
}

int TaskCPU::initImpl(std::function<void()> initHandle){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_init);
	m_InitHandle = initHandle;
	m_jobHandleQueue.push_back(m_InitHandle);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	return 0;
}

int TaskCPU::runImpl(std::function<void()> runHandle){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_run);
	m_RunHandle = runHandle;
	m_jobHandleQueue.push_back(runHandle);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	return 0;
}

int TaskCPU::execImpl(std::function<void()> execHandle){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_user_defined);
	m_ExecHandle = execHandle;
	m_jobHandleQueue.push_back(execHandle);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	return 0;
}

int TaskCPU::waitImpl(){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobDoneWait->reset(m_numLocalThreads);
	m_jobQueue.push_back(threadJobType::job_wait);
	m_jobHandleQueue.push_back(m_nullJobHandle);
	LCK1.unlock();
	m_jobExecCond.notify_all();
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] before wait!"<<std::endl;
#endif

	m_jobDoneWait->wait();

	return 0;
}

int TaskCPU::finishImpl(){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_finish);
	m_jobHandleQueue.push_back(m_nullJobHandle);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	// will wait all thread calling thread exit
	waitLocalThreadFinish();
}

void TaskCPU::threadSync(){
	// here is an synchronization point for task's local threads
	m_threadSync->count_down_and_wait();
}

void TaskCPU::threadSync(ThreadRank_t lrank){
	// here is an synchronization point for task's all threads
	if(m_numProcesses>1){
		if(lrank ==0){
	#ifdef USE_MPI_BASE
			MPI_Barrier(*m_commonTaskInfo->commPtr);
	#endif
		}
	}
	m_threadSync->count_down_and_wait();
}

void TaskCPU::threadExit(ThreadRank_t trank){
	// clear TSS data "m_taskInfo" defined in TaskManager
	TaskManager::setTaskInfo(nullptr);
	ThreadPrivateData *threadPrivateData = m_threadPrivateData->get();
	std::ofstream* m_threadOstream = threadPrivateData->threadOstream;
	if(m_threadOstream)
	{
		if(m_threadOstream->is_open())
		{
#ifdef USE_DEBUG_LOG
			PRINT_TIME_NOW(*m_threadOstream)
			(*m_threadOstream)<<"thread("<<trank<<") "<<std::this_thread::get_id()<<" exit!"<<std::endl;
#endif
			m_threadOstream->close();
		}
		delete m_threadOstream;
	}
	m_threadPrivateData->reset();  // clear TSS data

	//std::lock_guard<std::mutex> lock(m_activeLocalThreadMutex);
	m_activeLocalThreadMutex.lock();
	m_activeLocalThreadCount--;
	m_activeLocalThreadMutex.unlock();
	// notify main thread which is waiting for finish
	// only let last thread do notify
	if(m_activeLocalThreadCount == 0)
		//m_activeLocalThreadCond.notify_one(); // only main thread would wait for this
		m_activeLocalThreadCond.signal();

}

void TaskCPU::threadWait(){
	m_jobDoneWait->count_down();
}

bool TaskCPU::hasActiveLocalThread()
{
	/*
    std::lock_guard<std::mutex> lock(m_activeLocalThreadMutex);
    if(m_activeLocalThreadCount > 0)
        return true;
    else
        return false;
        */
	m_activeLocalThreadMutex.lock();
	bool ret = m_activeLocalThreadCount >0 ? true: false;
	m_activeLocalThreadMutex.unlock();
	return ret;
}

void TaskCPU::waitLocalThreadFinish()
{
	/*
    std::unique_lock<std::mutex> LCK(m_activeLocalThreadMutex);
    while(m_activeLocalThreadCount!=0)
    {
        m_activeLocalThreadCond.wait(LCK);
    }
    */
	m_activeLocalThreadMutex.lock();
	while(m_activeLocalThreadCount !=0){
		m_activeLocalThreadCond.wait(&m_activeLocalThreadMutex);
	}
}

}// end namespace iUtc
