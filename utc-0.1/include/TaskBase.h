#ifndef UTC_TASKBASE_H_
#define UTC_TASKBASE_H_

#include "UtcBasics.h"
#include "TaskInfo.h"
#include "UniqueExeTag.h"
#include "ProcList.h"

#include <map>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "boost/thread/tss.hpp"
#include "boost/thread/thread.hpp"



namespace iUtc{

enum class TaskType{
	unknown =0,
	cpu_task,
	gpu_task
};


class TaskBase{
public:
    std::string getName();

    TaskId_t getTaskId();

    TaskBase* getParentTask();
    TaskId_t getParentTaskId();

    TaskType getTaskType();

    //
    std::vector<ProcRank_t> getTaskRankList();

    int getNumProcesses();

    // convert thread global rank(among all processes of a task) to local rank
    // (the rank on current process)
    int toLocal(ThreadRank_t trank);

    //
    int getNumLocalThreads();
    std::vector<ThreadId_t> getLocalThreadList();

    int getNumTotalThreads();

    ProcRank_t getCurrentProcRank();
    ProcRank_t getProcRankOfThread(ThreadRank_t trank);

    ProcRank_t getMainResideProcess();

    // return if thread with rank value 'tRank' is on current process
    bool isLocal(ThreadRank_t tRank);

    ThreadRank_t getThreadRankById(ThreadId_t tid);

    static ThreadPrivateData* getThreadPrivateData();
    static void setThreadPrivateData(ThreadPrivateData* tpd);

    bool isActiveOnCurrentProcess();

    void display();

    //
	virtual bool hasActiveLocalThread();
	virtual void waitLocalThreadFinish();

    //
    virtual ~TaskBase();

    //
#if ENABLE_SCOPED_DATA
	internal_MPIWin *getTaskMpiWindow();
#endif
	std::map<int, int> *getProcWorldToTaskGroupMap();
	std::map<int, int> *getProcTaskGroupToWorldMap();


protected:
    std::string m_Name;

    TaskType m_TaskType;

    TaskId_t m_TaskId;

    TaskId_t m_ParentTaskId;

    int    m_numProcesses;

    // process related, for a task obj, in different process it may get
    // different value, depends on the ranklist of the task
    int    m_numLocalThreads;

    int    m_numTotalThreads;

    ProcRank_t m_processRank;    // the current running process's rank value

    ProcRank_t m_mainResideProcess;  // a main process rank that a task mapped to

    std::vector<ProcRank_t> m_TaskRankList;  //how task threads map to proc

    std::vector<ThreadId_t> m_LocalThreadList;

    std::map<ThreadId_t, ThreadRank_t> m_LocalThreadRegistry;

    std::map<ThreadRank_t, int> m_ThreadRank2Local;

    // output stream obj
    std::ofstream *m_procOstream;
    static boost::thread_specific_ptr<ThreadPrivateData> m_threadPrivateData;

    //
    void RegisterTask();

    // used for unique execution control
    UniqueExeTag *m_uniqueExeObjPtr;

    // used for bcast within task
    std::atomic<int> m_bcastAvailable;
    std::atomic<int> m_gatherAvailable;

    // can't use taskbase obj directly
    TaskBase();



    // an array to map world-mpi-rank to task-mpi-goup-rank
    std::map<int, int> m_worldRankToTaskGroupRank;
    std::map<int, int> m_taskGroupRankToWorldRank;

	//a mpi window used for implement global shared data object
#if ENABLE_SCOPED_DATA
	internal_MPIWin *m_taskMpiInternalWindow;
#endif
	long m_shmemSize;



};

}

#endif

