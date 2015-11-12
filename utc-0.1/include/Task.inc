namespace iUtc{

template<class T>
Task<T>::Task()
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	m_threadTerminateSignal = 0;
	m_calledRun = 0;
	m_calledInit = 0;

	m_threadReady2InitCounter =0;
	m_threadSyncInitCounterComing =0;
	m_threadSyncInitCounterLeaving =0;
	m_threadReady2RunCounter =0;


	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, RankList(1));
 }

template<class T>
Task<T>::Task(RankList rList)
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	m_threadTerminateSignal = 0;
	m_calledRun = 0;
	m_calledInit = 0;

	m_threadReady2InitCounter =0;
	m_threadSyncInitCounterComing =0;
	m_threadSyncInitCounterLeaving =0;
	m_threadReady2RunCounter =0;


	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, rList);
 }

template<class T>
Task<T>::Task( std::string name,  RankList rList)
:TaskBase()
 {
	m_taskThreads.clear();
	m_userTaskObjPtr = nullptr;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;

	m_threadTerminateSignal = 0;
	m_calledRun = 0;
	m_calledInit = 0;

	m_threadReady2InitCounter =0;
	m_threadSyncInitCounterComing =0;
	m_threadSyncInitCounterLeaving =0;
	m_threadReady2RunCounter =0;

	CreateTask(name, rList);
 }

template<class T>
Task<T>::~Task()
{
	if(m_numLocalThreads !=0)
	{
		if(m_calledInit != 1)
		{
			m_threadTerminateSignal = 1;
			initImpl();
		}
		else if(m_calledRun != 1)
		{
			m_threadTerminateSignal =1 ;
			runImpl();
		}
	}

	if(!m_taskBarrierObjPtr)
	{
		delete m_taskBarrierObjPtr;
		m_taskBarrierObjPtr=nullptr;
	}

	if(!m_userTaskObjPtr)
	{   // will call user task class's destructor
	    delete m_userTaskObjPtr;
	    m_userTaskObjPtr = nullptr;
	}

	for(auto& th : m_taskThreads)
	{
		if(th.joinable())
		{
			ThreadId id = th.get_id();
			th.join();
#ifdef USE_DEBUG_LOG
			PRINT_TIME_NOW(*m_procOstream)
			(*m_procOstream)<<"thread "<<id<<" join on "<<getpid()
					<<" for ["<<m_Name<<"]"<<std::endl;
#endif
		}
	}
	m_taskThreads.clear();

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
	m_calledInit = 1;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr);
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
	m_calledInit = 1;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1);
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
	m_calledInit = 1;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::init, m_userTaskObjPtr, arg1, arg2);
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
	m_calledRun = 1;
	// ask all thread to do task.run
	runImpl();

}

template<class T>
void Task<T>::waitTillDone()
{
    // no task threads on this process, just return
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
	// will wait all thread calling thread exit
    waitLocalThreadFinish();

#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] finish run!"<<std::endl;
#endif
}


template<class T>
int Task<T>::initImpl()
{
	//  tell all threads that they can execute task.init
	std::unique_lock<std::mutex> LCK(m_threadReady2InitMutex);
	m_threadReady2InitCounter =1;
	LCK.unlock();
	// if there are waiting threads, notify them to do task.init
	m_threadReady2InitCond.notify_all();

	// TODO: whether need to wait for all thread finishing task.init
	// wait for finish and return

	return 0;
}

template<class T>
int Task<T>::runImpl()
{
	// all threads can do task.run
	std::unique_lock<std::mutex> LCK(m_threadReady2RunMutex);
	m_threadReady2RunCounter=1;
	// no need unlock, will unlock implicitly at end of this code block
	//LCK.unlock();
	// notify waiting threads to do task.run
	m_threadReady2RunCond.notify_all();

	// TODO: whether need to wait for all thread finishing task.run

	return 0;
}

template<class T>
void Task<T>::threadImpl(ThreadRank trank, std::ofstream *output)
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
	taskInfoPtr->taskId = m_TaskId;
	taskInfoPtr->threadId = std::this_thread::get_id(); //same as m_LocalThreadList(local_rank)
	taskInfoPtr->barrierObjPtr = m_taskBarrierObjPtr;
#ifdef USE_MPI_BASE
	taskInfoPtr->commPtr = &m_taskComm;
	taskInfoPtr->mpigroupPtr = &m_taskmpiGroup;
#endif
	TaskManager::setTaskInfo(taskInfoPtr);

	ThreadPrivateData *threadPrivateData = new ThreadPrivateData();
	threadPrivateData->threadOstream = output;
	m_threadPrivateData.reset(threadPrivateData);

	//
	// do task.init
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


	// wait for all threads finishing init() call,
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
	m_userTaskObjPtr->run();

	// TODO: notify main thread that this thread finish run


	// thread exit
	threadExit(trank);
	return;
}

template<class T>
void Task<T>::threadExit(ThreadRank trank)
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
void Task<T>::CreateTask(const std::string name, const RankList rList)
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
	m_ParentTaskId = TaskManager::getCurrentTaskId();  // here we will get root task's id
	rList.getRankListVector(m_TaskRankList);
	m_numTotalThreads = m_TaskRankList.size();
	int currentProcRank = root->getCurrentProcRank();
	m_processRank = currentProcRank;
	int nProcWorld = root->getNumProcesses();
	std::vector<std::vector<ThreadRank>> plist(nProcWorld);
	//std::vector<ThreadRank> plist[nProcWorld];
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
void Task<T>::LaunchThreads(std::vector<ThreadRank> &tRankList)
{
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
	ThreadId id;
	for(int i=0; i<m_numLocalThreads; i++)
	{
		trank=tRankList[i];
#ifdef USE_DEBUG_LOG
		boost::filesystem::path log_path("./log");
		if(!boost::filesystem::exists(log_path))
			boost::filesystem::create_directory(log_path);
		std::string filename = "./log/";
		filename.append(m_Name);
		filename.append("-thread");
		filename.append(std::to_string(trank));
		filename.append(".log");
		std::ofstream *output = new std::ofstream(filename);
		filename.clear();
#else
		std::ofstream *output = nullptr;
#endif

		m_ThreadRank2Local.insert(std::pair<ThreadRank, int>(trank, i));
		// crteate a thread
		m_taskThreads.push_back(std::thread(&Task<T>::threadImpl, this, trank, output));
		// record some info
		id = m_taskThreads.back().get_id();
		m_LocalThreadList.push_back(id);
		m_LocalThreadRegistry.insert(std::pair<ThreadId, ThreadRank>(id, trank));
	}

#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<m_numLocalThreads<<" threads for ["<<m_Name<<"] launched on proc "<<m_processRank
			<<std::endl;
#endif
}




} //namespace iUtc
