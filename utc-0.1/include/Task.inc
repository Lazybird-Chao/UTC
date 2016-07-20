namespace iUtc{

template<class T>
Task<T>::Task()
:TaskBase()
 {
	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = TaskType::cpu_task;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));

	int numProcess = TaskManager::getRootTask()->getNumProcesses();
	std::vector<int> rlist;
	for(int i=0;i<numProcess;i++)
		rlist.push_back(i);
	CreateTask(name, ProcList(rlist), TaskType::cpu_task);
 }

template<class T>
Task<T>::Task(ProcList rList)
:TaskBase()
 {
	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = TaskType::cpu_task;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, rList, TaskType::cpu_task);
 }

template<class T>
Task<T>::Task(ProcList rList, TaskType tType)
:TaskBase()
 {

	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = tType;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, rList, tType);
 }

template<class T>
Task<T>::Task(T* userTaskObj, ProcList rList, TaskType tType)
:TaskBase()
 {
	m_userTaskObjPtr = userTaskObj;
	m_userSupplyObj = true;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = tType;

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));
	CreateTask(name, rList, tType);
 }

template<class T>
Task<T>::Task(std::string name)
:TaskBase()
 {

	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = TaskType::cpu_task;

	int numProcess = TaskManager::getRootTask()->getNumProcesses();
	std::vector<int> rlist;
	for(int i=0;i<numProcess;i++)
		rlist.push_back(i);
	CreateTask(name, rlist, TaskType::cpu_task);
 }

template<class T>
Task<T>::Task( std::string name,  ProcList rList)
:TaskBase()
 {

	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = TaskType::cpu_task;

	CreateTask(name, rList, TaskType::cpu_task);
 }

template<class T>
Task<T>::Task(std::string name, ProcList rList, TaskType tType)
:TaskBase()
 {

	m_userTaskObjPtr = nullptr;
	m_userSupplyObj = false;
	m_procOstream = nullptr;
	m_taskBarrierObjPtr = nullptr;
	m_taskSpinBarrierObjPtr = nullptr;
	callTaskFinish = false;
	m_realTaskInstance= nullptr;
	m_TaskType = tType;

	CreateTask(name, rList, tType);
 }

template<class T>
Task<T>::~Task()
{
	if(m_numLocalThreads !=0)
	{
		//std::cout<<ERROR_LINE<<std::endl;
		if(!callTaskFinish){
			m_realTaskInstance->finishImpl();
			callTaskFinish = true;
#ifdef USE_MPI_BASE
			MPI_Barrier(m_taskComm);
#endif
#ifdef USE_DEBUG_LOG
	PRINT_TIME_NOW(*m_procOstream)
	(*m_procOstream)<<"["<<m_Name<<"] finish run!"<<std::endl;
#endif
		}

		delete m_realTaskInstance;

	}

	if(m_taskBarrierObjPtr!=nullptr)
	{
		delete m_taskBarrierObjPtr;
		m_taskBarrierObjPtr=nullptr;
		delete m_taskSpinBarrierObjPtr;
		m_taskSpinBarrierObjPtr=nullptr;
	}

	if(m_userSupplyObj == false && m_userTaskObjPtr)
	{   // will call user task class's destructor
			delete m_userTaskObjPtr;
	}
	m_userTaskObjPtr = nullptr;


	TaskManager::unregisterTask(this, m_TaskId);

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
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr);

	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);

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
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);

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
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);

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
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);

	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
	return;
}
template<class T>
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
void Task<T>::init(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
{
    if(m_numLocalThreads ==0)
    	return;
	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);
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
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}

template<class T>
template<typename T1>
void Task<T>::run(T1 arg1)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr, arg1);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}


template<class T>
void Task<T>::wait(){
	if(m_numLocalThreads ==0)
	{
		return;
	}

	m_realTaskInstance->waitImpl();

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

    m_realTaskInstance->finishImpl();
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

	m_realTaskInstance->execImpl(m_userTaskExecHandle);
	return;
}
template<class T>
template<typename T1>
void Task<T>::exec(void (T::*user_fun)(T1), T1 arg1){
	if(m_numLocalThreads ==0)
		return;
	//m_userTaskExecHandle = std::bind(static_cast<void(T::*)(T1)>(user_fun), m_userTaskObjPtr, arg1);
	m_userTaskExecHandle = std::bind(user_fun, m_userTaskObjPtr, arg1);
	m_realTaskInstance->execImpl(m_userTaskExecHandle);
	return;
}



template<class T>
void Task<T>::CreateTask(const std::string name, const ProcList rList, TaskType tType)
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
			std::cerr<<"nProcWorld: "<<nProcWorld<<"  but want: "<<m_TaskRankList[i]<<std::endl;
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
		if(m_userTaskObjPtr == nullptr){
			// will call user task class's constructor
			m_userTaskObjPtr = new T;
		}

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

	if(m_numLocalThreads !=0){
		// create common taskInfo and threadPrivateData structure
		m_commonTaskInfo.pRank = m_processRank;
		m_commonTaskInfo.parentTaskId = m_ParentTaskId;
		m_commonTaskInfo.taskId = m_TaskId;
		m_commonTaskInfo.barrierObjPtr = m_taskBarrierObjPtr;
		m_commonTaskInfo.spinBarrierObjPtr = m_taskSpinBarrierObjPtr;
		#ifdef USE_MPI_BASE
		m_commonTaskInfo.commPtr = &m_taskComm;
		m_commonTaskInfo.mpigroupPtr = &m_taskmpiGroup;
		#endif

		m_commonThreadPrivateData.taskUniqueExeTagObj = m_uniqueExeObjPtr;
		m_commonThreadPrivateData.bcastAvailable = &m_bcastAvailable;
		m_commonThreadPrivateData.gatherAvailable = &m_gatherAvailable;

	}
	//std::cout<<ERROR_LINE<<std::endl;
	if(m_numLocalThreads!=0)  // there are threads need to be created on current process
	{
		if(tType==TaskType::cpu_task){
			m_realTaskInstance = new TaskCPU(m_numLocalThreads,
											m_processRank,
											m_numProcesses,
											m_numTotalThreads,
											plist[currentProcRank],
											&m_LocalThreadList,
											&m_LocalThreadRegistry,
											&m_ThreadRank2Local,
											m_procOstream,
											&m_commonTaskInfo,
											&m_commonThreadPrivateData,
											&m_threadPrivateData,
											m_userTaskObjPtr);
		}
		else{
			std:cerr<<"Error, unsupported task type!!!\n";
			exit(1);
		}

		m_realTaskInstance->launchThreads();
		//std::cout<<ERROR_LINE<<std::endl;
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
bool Task<T>::hasActiveLocalThread(){
	if(m_realTaskInstance)
		return m_realTaskInstance->hasActiveLocalThread();
	else
		return false;
}

template<class T>
void Task<T>::waitLocalThreadFinish(){
	if(m_realTaskInstance){
		m_realTaskInstance->waitLocalThreadFinish();
	}
	return;
}

} //namespace iUtc
