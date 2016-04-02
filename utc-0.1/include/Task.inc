namespace iUtc{

template<class T>
Task<T>::Task()
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));

	int numProcess = TaskManager::getRootTask()->getNumProcesses();
	std::vector<int> rlist;
	for(int i=0;i<numProcess;i++)
		rlist.push_back(i);
	CreateTask(name, ProcList(rlist));
 }

template<class T>
Task<T>::Task(ProcList rList)
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, rList);
 }

template<class T>
Task<T>::Task(std::string name)
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	int numProcess = TaskManager::getRootTask()->getNumProcesses();
	std::vector<int> rlist;
	for(int i=0;i<numProcess;i++)
		rlist.push_back(i);
	CreateTask(name, rlist);
 }

template<class T>
Task<T>::Task( std::string name,  ProcList rList)
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	CreateTask(name, rList);
 }

template<class T>
Task<T>::~Task()
{
	if(m_numLocalThreads !=0)
	{
		if(!callTaskFinish){
			std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
			m_jobQueue.push_back(threadJobType::job_finish);
			LCK1.unlock();
			m_jobExecCond.notify_all();
			// will wait all thread calling thread exit
			waitLocalThreadFinish();
			callTaskFinish = true;
#ifdef USE_MPI_BASE
			MPI_Barrier(m_taskComm);
#endif
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] finish run!"<<std::endl;
#endif
		}


		for(auto& th : m_taskThreads)
		{
			if(th.joinable())
			{
				ThreadId_t id = th.get_id();
				th.join();
#ifdef USE_DEBUG_LOG
			PRINT_TIME_NOW(*m_procOstream)
			(*m_procOstream)<<"thread "<<id<<" join on "<<getpid()
					<<" for ["<<m_Name<<"]"<<std::endl;
#endif
			}
		}
		m_taskThreads.clear();


		delete m_threadSync;
		delete m_jobDoneWait;
		delete m_threadJobIdx;
		m_jobQueue.clear();
		m_jobHandleQueue.clear();
	}

	if(m_taskBarrierObjPtr!=nullptr)
	{
		delete m_taskBarrierObjPtr;
		m_taskBarrierObjPtr=nullptr;
		delete m_taskSpinBarrierObjPtr;
		m_taskSpinBarrierObjPtr=nullptr;
	}

	if(m_userTaskObjPtr!=nullptr)
	{   // will call user task class's destructor
		delete m_userTaskObjPtr;
		m_userTaskObjPtr = nullptr;
	}


	TaskManager::unregisterTask(this, m_TaskId);

	//if(m_numLocalThreads !=0)
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_procOstream)
		(*m_procOstream)<<"["<<m_Name<<"] destroyed on proc "<<m_processRank<<" !!!"<<std::endl;
#endif
}

template<class T>
void Task<T>::init()
{
    // no task threads on this process, just return
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }

	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();

	return;
}
template<class T>
template<typename T1>
void Task<T>::init(T1 arg1)
{
    // no task threads on this process, just return
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }

	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();

	return;
}
template<class T>
template<typename T1, typename T2>
void Task<T>::init(T1 arg1, T2 arg2)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }

	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();

	return;
}
template<class T>
template<typename T1, typename T2, typename T3>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }

	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();

	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
	m_jobHandleQueue.push_back(m_userTaskInitFunctionHandle);
	// ask all threads to do task.init
	initImpl();
	return;
}



template<class T>
void Task<T>::run()
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::run, m_userTaskObjPtr);
    m_jobHandleQueue.push_back(m_userTaskRunFunctionHandle);
    runImpl();
    return;
}

template<class T>
void Task<T>::wait(){
	if(m_numLocalThreads ==0)
	{
		return;
	}
	m_jobHandleQueue.push_back(m_nullJobHandle);
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobDoneWait->reset(m_numLocalThreads);
	m_jobQueue.push_back(threadJobType::job_wait);
	LCK1.unlock();
	m_jobExecCond.notify_all();

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] before wait!"<<std::endl;
#endif

	m_jobDoneWait->wait();

    // do barrier across all processes that runing this task
#ifdef USE_MPI_BASE
	MPI_Barrier(m_taskComm);
#endif

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] after wait!"<<std::endl;
#endif
	return;
}


template<class T>
void Task<T>::finish()
{
    // no task threads on this process, just return
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
#ifdef USE_DEBUG_ASSERT
    assert(callTaskFinish==false);
#endif
    m_jobHandleQueue.push_back(m_nullJobHandle);
    std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_finish);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	// will wait all thread calling thread exit
    waitLocalThreadFinish();
    callTaskFinish = true;

    // do barrier across all processes that runing this task
#ifdef USE_MPI_BASE
     MPI_Barrier(m_taskComm);
#endif

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] finish run!"<<std::endl;
#endif
}

template<class T>
void Task<T>::exec(void (T::*user_fun)()){
	if(m_numLocalThreads ==0)
		return;
	m_userTaskExecHandle = std::bind(user_fun, m_userTaskObjPtr);
	m_jobHandleQueue.push_back(m_userTaskExecHandle);
	execImpl();
	return;
}
template<class T>
template<typename T1>
void Task<T>::exec(void (T::*user_fun)(T1), T1 arg1){
	if(m_numLocalThreads ==0)
		return;
	//m_userTaskExecHandle = std::bind(static_cast<void(T::*)(T1)>(user_fun), m_userTaskObjPtr, arg1);
	m_userTaskExecHandle = std::bind(user_fun, m_userTaskObjPtr, arg1);
	m_jobHandleQueue.push_back(m_userTaskExecHandle);
	execImpl();
	return;
}

template<class T>
int Task<T>::execImpl(){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_user_defined);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	return 0;
}

template<class T>
int Task<T>::initImpl()
{
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_init);
	LCK1.unlock();
	m_jobExecCond.notify_all();
	return 0;
}

template<class T>
int Task<T>::runImpl(){
	std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
	m_jobQueue.push_back(threadJobType::job_run);
	LCK1.unlock();
	m_jobExecCond.notify_all();
}


template<class T>
void Task<T>::threadImpl(ThreadRank_t trank, ThreadRank_t lrank, std::ofstream *output)
{
	//
	std::ofstream *m_threadOstream = output;
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_threadOstream)
	(*m_threadOstream)<<"thread("<<trank<<") "<<std::this_thread::get_id()<<" start up on "
			<<getpid()<<"..."<<std::endl;
#endif

	// create task info in this thread TSS
	TaskInfo* taskInfoPtr = new TaskInfo();
	taskInfoPtr->pRank = m_processRank;
	taskInfoPtr->parentTaskId = m_ParentTaskId;
	taskInfoPtr->tRank = trank;
	taskInfoPtr->lRank = lrank;
	taskInfoPtr->taskId = m_TaskId;
	taskInfoPtr->threadId = std::this_thread::get_id(); //same as m_LocalThreadList(local_rank)
	taskInfoPtr->barrierObjPtr = m_taskBarrierObjPtr;
	taskInfoPtr->spinBarrierObjPtr = m_taskSpinBarrierObjPtr;
#ifdef USE_MPI_BASE
	taskInfoPtr->commPtr = &m_taskComm;
	taskInfoPtr->mpigroupPtr = &m_taskmpiGroup;
#endif
	TaskManager::setTaskInfo(taskInfoPtr);

	ThreadPrivateData *threadPrivateData = new ThreadPrivateData();
	threadPrivateData->threadOstream = output;
	threadPrivateData->taskUniqueExeTagObj = m_uniqueExeObjPtr;
	threadPrivateData->bcastAvailable = &m_bcastAvailable;
	threadPrivateData->gatherAvailable = &m_gatherAvailable;
	m_threadPrivateData.reset(threadPrivateData);

	//
	/*// do task.init
	std::unique_lock<std::mutex> LCK1(m_threadReady2InitMutex);
	m_threadReady2InitCond.wait(LCK1,
			[=](){return m_threadReady2InitCounter ==1;});
	if(m_threadTerminateSignal == 1)
	{	// wake up by terminate call, just end this thread
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		(*m_threadOstream)<<"thread "<<std::this_thread::get_id()<<" get terminated on "
				<<getpid()<<"!"<<std::endl;
#endif
		threadExit(trank);
		return;
	}
	LCK1.unlock();


	// call user's task.init() with user task obj
	// m_userTaskObjPtr->init();
	m_userTaskInitFunctionHandle();

	// TODO: notifying main thread that this user task.init is finished


	// wait for all local threads finishing init() call,
	// here is an synchronization point for task threads
	std::unique_lock<std::mutex> LCK_initsync(m_threadSyncInitMutex);
	while(m_threadSyncInitCounterComing < m_numLocalThreads)
	{
		m_threadSyncInitCounterComing++;
		if(m_threadSyncInitCounterComing != m_numLocalThreads)
		{
			// not the last coming here thread, increase counter and wait
			m_threadSyncInitCond.wait(LCK_initsync);

		}
		else
		{	// last coming here thread
			m_threadSyncInitCond.notify_all();
		}
	}
	m_threadSyncInitCounterLeaving++;   // record one thread leave the sync point
	if(m_threadSyncInitCounterLeaving==m_numLocalThreads)
	{
		// the last thread that finish sync, reset sync counter pair
		m_threadSyncInitCounterComing=0;
		m_threadSyncInitCounterLeaving =0;
	}
	LCK_initsync.unlock();


	// do task.run
	std::unique_lock<std::mutex> LCK2(m_threadReady2RunMutex);
	m_threadReady2RunCond.wait(LCK2,
				[=](){return m_threadReady2RunCounter ==1;});
	if(m_threadTerminateSignal == 1)
	{	// wake up by terminate call, just end this thread
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_threadOstream)
		(*m_threadOstream)<<"thread "<<std::this_thread::get_id()<<" get terminated on"
						<<getpid()<<std::endl;
#endif
		threadExit(trank);
		return;
	}
	// wake up by task.run call
	LCK2.unlock();

	//call user's task.run()
	//m_userTaskObjPtr->run();
	m_userTaskRunFunctionHandle();

	// TODO: notify main thread that this thread finish run*/

	while(1){
		std::unique_lock<std::mutex> LCK1(m_jobExecMutex);
		m_jobExecCond.wait(LCK1,
				[=](){return m_jobQueue.size()> m_threadJobIdx[lrank];});
		int tmpJobHandle = m_jobQueue[m_threadJobIdx[lrank]];
		LCK1.unlock();

		switch(tmpJobHandle){
		case threadJobType::job_init:
			m_userTaskInitFunctionHandle();
			threadSync(lrank);
			break;
		case threadJobType::job_run:
			m_userTaskRunFunctionHandle();
			//threadSync();
			break;
		case threadJobType::job_finish:
			//threadExit(trank);
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
			//m_userTaskExecHandle();
			m_jobHandleQueue[m_threadJobIdx[lrank]]();
			threadSync();
			break;
		default:
			std::cerr<<"Error! Undefined Task job type!"<<std::endl;
			break;
		}
		m_threadJobIdx[lrank]++;
		if(tmpJobHandle == threadJobType::job_finish)
			break;
	}
	// do thread exit clear work at this point
	threadExit(trank);
	return;
}

template<class T>
void Task<T>::threadWait(){
	m_jobDoneWait->count_down();

}


template<class T>
void Task<T>::threadSync(){
	// here is an synchronization point for task's local threads
	m_threadSync->count_down_and_wait();
}
template<class T>
void Task<T>::threadSync(ThreadRank_t lrank ){
	// here is an synchronization point for task's all threads
	m_threadSync->count_down_and_wait();
	if(lrank ==0){
#ifdef USE_MPI_BASE
		MPI_Barrier(m_taskComm);
#endif
	}
    m_threadSync->count_down_and_wait();
}

template<class T>
void Task<T>::threadExit(ThreadRank_t trank)
{
	// clear TSS data "m_taskInfo" defined in TaskManager
	TaskManager::setTaskInfo(nullptr);

	ThreadPrivateData *threadPrivateData = m_threadPrivateData.get();
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
	m_threadPrivateData.reset();  // clear TSS data

	// reduce active thread count
    std::lock_guard<std::mutex> lock(m_activeLocalThreadMutex);
    m_activeLocalThreadCount--;
    // notify main thread which is waiting for finish
    // only let last thread do notify
    if(m_activeLocalThreadCount == 0)
        m_activeLocalThreadCond.notify_one(); // only main thread would wait for this
}

template<class T>
void Task<T>::CreateTask(const std::string name, const ProcList rList)
{
	RootTask *root = TaskManager::getRootTask();
	m_procOstream = root->getProcOstream();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"start creating ["<<name<<"] ..."<<std::endl;
#endif
	// init this new task obj's info, be careful this task obj is not root task,
	// but current program still running in main thread(or root task thread)
	m_Name = name;
	m_TaskId = TaskManager::getNewTaskId();
	m_ParentTaskId = TaskManager::getCurrentTaskId();  // in main program, it will get root task's id
	rList.getProcListVector(m_TaskRankList);
	m_numTotalThreads = m_TaskRankList.size();
	int currentProcRank = root->getCurrentProcRank();
	m_processRank = currentProcRank;
	int nProcWorld = root->getNumProcesses();
	std::vector<std::vector<ThreadRank_t>> plist(nProcWorld);
	for(int i=0; i<m_numTotalThreads; i++)
	{
		if(m_TaskRankList[i]>nProcWorld-1)
		{
			std::cerr<<"\n*****Error, Task ["<<m_Name<<"] mapped to non-existed processes!*****\n"<<std::endl;
			std::exit(1);
		}
		plist[m_TaskRankList[i]].push_back(i);
	}
	m_numProcesses = 0;
	m_numLocalThreads = 0;
	int *inc_rank = new int[nProcWorld];
	for(int i=0; i<nProcWorld; i++)
	{
		if(plist[i].size() != 0)
		{
			inc_rank[m_numProcesses]= i;
			m_numProcesses++;
		}
		if(currentProcRank == i)	// task has threads mapped to current process
		{
			m_numLocalThreads = plist[i].size();
		}
	}
	//the first process is set to be the main reside process for a task
	m_mainResideProcess = inc_rank[0];
#ifdef USE_MPI_BASE
	MPI_Group_incl(*(root->getWorldGroup()), m_numProcesses, inc_rank, &m_taskmpiGroup);
	MPI_Comm_create(*(root->getWorldComm()), m_taskmpiGroup, &m_taskComm);
#endif

	// create obj of user defined task class"T". With this design all threads in
	// one process will share this user taskobj, the data members of the this obj
	// will be accessed by all threads. Also each processes will has a same copy
	// of this user taskobj.
#ifdef USE_DEBUG_ASSERT
	assert(m_userTaskObjPtr == nullptr);
#endif
	if(m_numLocalThreads !=0)
	{
	    // will call user task class's constructor
	    m_userTaskObjPtr = new T;
	}

	if(m_numLocalThreads!=0)
	{
		// create the barrier object in a task
		m_taskBarrierObjPtr = new Barrier(m_numLocalThreads, m_TaskId, &m_taskComm);
		m_taskSpinBarrierObjPtr = new SpinBarrier(m_numLocalThreads);
	}

	if(m_numLocalThreads !=0){
		// init unique execution control info
		m_uniqueExeObjPtr = new UniqueExeTag(m_numLocalThreads, 32);
	}

	if(m_numLocalThreads!=0)  // there are threads need to be created on current process
	{
		LaunchThreads(plist[currentProcRank]);
	}

	// all task threads have launched, finish task creation
	// register current task obj in taskmanager
	RegisterTask();
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"finish creating ["<<name<<"]"<<std::endl;
#endif

}

template<class T>
void Task<T>::LaunchThreads(std::vector<ThreadRank_t> &tRankList)
{
	// do some constructor things that related to task-threads
	m_threadSync = new boost::barrier(m_numLocalThreads);
	m_jobDoneWait = new boost::latch(m_numLocalThreads);
	m_jobQueue.clear();
	m_jobHandleQueue.clear();
	m_threadJobIdx = new int[m_numLocalThreads];
	for(int i=0;i<m_numLocalThreads;i++)
		m_threadJobIdx[i]=0;
	callTaskFinish = false;

	////
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"start launching threads for ["<<m_Name<<"] on proc "<<m_processRank
			<<"..."<<std::endl;
#endif
	//
	std::unique_lock<std::mutex> LCK(m_activeLocalThreadMutex);
	m_activeLocalThreadCount = m_numLocalThreads;
	LCK.unlock();
	int trank=0;
	ThreadId_t id;
	for(int i=0; i<m_numLocalThreads; i++)
	{
		trank=tRankList[i];
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

		m_ThreadRank2Local.insert(std::pair<ThreadRank_t, int>(trank, i));
		// crteate a thread
		m_taskThreads.push_back(std::thread(&Task<T>::threadImpl, this, trank, i, output));
		//m_taskThreads[i].detach();
		// record some info
		id = m_taskThreads.back().get_id();
		m_LocalThreadList.push_back(id);
		m_LocalThreadRegistry.insert(std::pair<ThreadId_t, ThreadRank_t>(id, trank));
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<m_numLocalThreads<<" threads for ["<<m_Name<<"] launched on proc "<<m_processRank
			<<std::endl;
#endif
}




} //namespace iUtc
