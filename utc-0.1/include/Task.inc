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
Task<T>::Task(std::string name, ProcList rList, TaskType tType, long shmemSize)
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

#if ENABLE_SCOPED_DATA
	m_shmemSize = shmemSize;
#endif

	CreateTask(name, rList, tType);
 }

template<class T>
Task<T>::Task(ProcList rList, TaskType tType, long shmemSize)
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

#if ENABLE_SCOPED_DATA
	m_shmemSize = shmemSize;
#endif

	int ntasks = TaskManager::getNumTasks();
	std::string name = "Task";
	name.append(std::to_string(ntasks));

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
		//std::cout<<m_Name<<" finish "<<std::endl;
		delete m_realTaskInstance;
#ifdef SHOW_DEBUG
std::cout<<ERROR_LINE<<"Base task object destructed on required process."<<m_processRank<<std::endl;
#endif
	}

	if(m_taskBarrierObjPtr!=nullptr)
	{
		delete m_taskBarrierObjPtr;
		m_taskBarrierObjPtr=nullptr;
		delete m_taskSpinBarrierObjPtr;
		m_taskSpinBarrierObjPtr=nullptr;
		delete m_taskFastBarrierObjPtr;
		m_taskFastBarrierObjPtr = nullptr;
	}

	/*
	 * we should only delete user task object on processes that
	 * this task running on
	 */
	if(m_userSupplyObj == false && m_userTaskObjPtr)
	{   // will call user task class's destructor
			delete m_userTaskObjPtr;
	}
	m_userTaskObjPtr = nullptr;

	//
#if ENABLE_SCOPED_DATA
#ifdef USE_INTERNALSHMEM
	/*
	 * task related mpi window should be deleted after user object
	 */
	if(m_numLocalThreads !=0){
		delete m_taskMpiInternalWindow;
	}
#endif
#endif

	TaskManager::unregisterTask(this, m_TaskId);

#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*m_procOstream)
		(*m_procOstream)<<"["<<m_Name<<"] destroyed on proc "<<m_processRank<<" !!!"<<std::endl;
#endif
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"end task destructor process rank "<<m_processRank<<std::endl;
#endif
}

template<class T>
template<typename... Args>
void Task<T>::init(Args... args)
{
    // no task threads on this process, just return
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }

	//set up user task's init function
	m_userTaskInitFunctionHandle = std::bind(&T::initImpl, m_userTaskObjPtr, args...);

	m_realTaskInstance->initImpl(m_userTaskInitFunctionHandle);

	return;
}

/*
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
*/


template<class T>
template <typename... Args>
void Task<T>::run(Args... args)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr, args...);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}

/*
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
template<typename T1, typename T2>
void Task<T>::run(T1 arg1, T2 arg2)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr, arg1, arg2);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}

template<class T>
template<typename T1, typename T2, typename T3>
void Task<T>::run(T1 arg1, T2 arg2, T3 arg3)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr, arg1, arg2, arg3);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}

template<class T>
template<typename T1, typename T2, typename T3, typename T4>
void Task<T>::run(T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
    // no task threads on this process, just return.
    // same as m_userTaskObjPtr==nullptr
    if(m_numLocalThreads ==0)
    {
        return;
    }
    m_userTaskRunFunctionHandle = std::bind(&T::runImpl, m_userTaskObjPtr, arg1, arg2, arg3, arg4);
    m_realTaskInstance->runImpl(m_userTaskRunFunctionHandle);
    return;
}
*/

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
template<typename... Args>
void Task<T>::exec(void (T::*user_fun)(Args...), Args... args){
	if(m_numLocalThreads ==0)
		return;
	m_userTaskExecHandle = std::bind(user_fun, m_userTaskObjPtr, args...);

	m_realTaskInstance->execImpl(m_userTaskExecHandle);
	return;
}

/*
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
*/



/*
 * TODO: we also need to update the task artifact(CPU task or GPU task)
 * update the userTaskObj there!!!
 */
template<class T>
int Task<T>::setUserTaskObj(T &userTaskObj){
	int ret = 0;
	if(m_userTaskObjPtr != nullptr){
		delete m_userTaskObjPtr;
		ret = 1;
	}
	m_userTaskObjPtr = &userTaskObj;
	m_userSupplyObj  = true;
	return ret;
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

	/*
	 * for each task, we careat a new communicator, so the process rank in
	 * this m_taskComm is different from the world_rank value.
	 * we record the map from world_rank to task_mpi_group_rank
	 */
	int *rankInGroup = new int[m_numProcesses];
	MPI_Group_translate_ranks(*(root->getWorldGroup()), m_numProcesses, inc_rank,
			m_taskmpiGroup, rankInGroup);
	for(int i=0; i<m_numProcesses; i++){
		m_worldRankToTaskGroupRank.insert(std::pair<int,int>(inc_rank[i], rankInGroup[i]));
		m_taskGroupRankToWorldRank.insert(std::pair<int,int>(rankInGroup[i], inc_rank[i]));
	}

#endif

#if ENABLE_SCOPED_DATA
#ifdef USE_INTERNALSHMEM
	/*
	 * Noticing: we created base task object on every processes, but we only launch
	 * threads on those processes that a task required, so only on those processes,
	 * the base task object has meaning, also we will create user task object on those
	 * processes. So all kinds of data that are useful with a task should be created
	 * in those base task objects, like this window object. If we create some data
	 * on every process, but it's related to a task's required processes, it may cause
	 * fatal error when other process is running or finish.
	 */
	if(m_numLocalThreads != 0){
		if(m_shmemSize ==0)
			m_taskMpiInternalWindow = new internal_MPIWin(&m_taskComm, SHMEM_SIZE, 0);
		else
			m_taskMpiInternalWindow = new internal_MPIWin(&m_taskComm, m_shmemSize, 0);

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"internal mpiwin created. "<<
			"["<<m_processRank<<" "<<m_worldRankToTaskGroupRank[m_processRank]
			 <<"]"<<std::endl;
#endif
	}
#endif
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
	//std::cout<<ERROR_LINE<<std::endl;
	if(m_numLocalThreads!=0)
	{
		// create the barrier object in a task
		m_taskBarrierObjPtr = new Barrier(m_numLocalThreads, m_TaskId, &m_taskComm);
		m_taskSpinBarrierObjPtr = new SpinBarrier(m_numLocalThreads);
		m_taskFastBarrierObjPtr = new FastBarrier(m_numLocalThreads);
	}

	if(m_numLocalThreads !=0){
		// init unique execution control info
		m_uniqueExeObjPtr = new UniqueExeTag(m_numLocalThreads, 32);
	}

	if(m_numLocalThreads !=0){
		// create common taskInfo and threadPrivateData structure
		m_commonTaskInfo.pRank = m_processRank;
		m_commonTaskInfo.numWorldProcesses = nProcWorld;
		m_commonTaskInfo.procGroupRank = m_worldRankToTaskGroupRank[m_processRank];
		m_commonTaskInfo.numGroupProcesses = m_numProcesses;
		m_commonTaskInfo.parentTaskId = m_ParentTaskId;
		m_commonTaskInfo.taskId = m_TaskId;
		m_commonTaskInfo.barrierObjPtr = m_taskBarrierObjPtr;
		m_commonTaskInfo.spinBarrierObjPtr = m_taskSpinBarrierObjPtr;
		m_commonTaskInfo.fastBarrierObjPtr = m_taskFastBarrierObjPtr;
		#ifdef USE_MPI_BASE
		m_commonTaskInfo.commPtr = &m_taskComm;
		m_commonTaskInfo.mpigroupPtr = &m_taskmpiGroup;

		m_commonTaskInfo.worldCommPtr = root->getWorldComm();
		m_commonTaskInfo.worldGroupPtr = root->getWorldGroup();
		m_commonTaskInfo.worldRankToGrouprRank = &m_worldRankToTaskGroupRank;
		m_commonTaskInfo.groupRankToWorldRank = &m_taskGroupRankToWorldRank;
		#endif

		#if ENABLE_SCOPED_DATA
		#ifdef USE_INTERNALSHMEM
		m_commonTaskInfo.taskMpiInternalWindow = m_taskMpiInternalWindow;
		#endif
		#endif

		m_commonThreadPrivateData.taskUniqueExeTagObj = m_uniqueExeObjPtr;
		m_commonThreadPrivateData.bcastAvailable = &m_bcastAvailable;
		m_commonThreadPrivateData.gatherAvailable = &m_gatherAvailable;

	}
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"pre create internal cpu/gpu taskobj."<<std::endl;
#endif
	if(m_numLocalThreads!=0)  // there are threads need to be created on current process
	{
		if(tType==TaskType::cpu_task){
			m_realTaskInstance = new TaskCPU(tType,
											m_numLocalThreads,
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
#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"cpu task obj created."<<std::endl;
#endif
		}
		else if(tType==TaskType::gpu_task){
#if ENABLE_GPU_TASK
			m_realTaskInstance = new TaskGPU(tType,
											m_numLocalThreads,
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
											m_userTaskObjPtr
											);
#else
			m_realTaskInstance = nullptr;
			std::cerr<<"Error, gpu task unabled!!!\n";
#endif

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"gpu task created."<<std::endl;
#endif
		}
		else{
			std:cerr<<"Error, unsupported task type!!!\n";
			exit(1);
		}

		//m_realTaskInstance->launchThreads();
		//std::cout<<ERROR_LINE<<std::endl;
	}

	// register current task obj in taskmanager
	RegisterTask();

	// move to here, after task Register, to avoid possible bug
	if(m_numLocalThreads != 0){
		m_realTaskInstance->launchThreads();

		MPI_Barrier(m_taskComm);

#ifdef SHOW_DEBUG
	std::cout<<ERROR_LINE<<"task threads launched."<<std::endl;
#endif
	}

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
