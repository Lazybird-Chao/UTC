#ifndef UTC_TASKBASE_H_
#define UTC_TASKBASE_H_

#include "UtcBasics.h"
#include "TaskInfo.h"


#include <map>
#include <vector>
#include <string>
#include "boost/thread/tss.hpp"
#include "boost/thread/thread.hpp"
#include <mutex>
#include <condition_variable>
#include "ProcList.h"

namespace iUtc{

class TaskBase{
public:
    std::string getName();

    TaskId_t getTaskId();

    TaskBase* getParentTask();
    TaskId_t getParentTaskId();

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

    ProcRank_t getMainResideProcess();

    // return if thread with rank value 'tRank' is on current process
    bool isLocal(ThreadRank_t tRank);

    ThreadRank_t getThreadRankById(ThreadId_t tid);

    static ThreadPrivateData* getThreadPrivateData();
    static void setThreadPrivateData(ThreadPrivateData* tpd);

    bool isActiveOnCurrentProcess();

    bool hasActiveLocalThread();
    void waitLocalThreadFinish();

    void display();

    //
    virtual ~TaskBase();

protected:
    std::string m_Name;

    TaskId_t m_TaskId;

    TaskId_t m_ParentTaskId;

    int    m_numProcesses;

    // process related, for a task obj, in different process it may get
    // different value, depends on the ranklist of the task
    int    m_numLocalThreads;

    int    m_numTotalThreads;

    ProcRank_t m_processRank;    // the current running process's rank value

    ProcRank_t m_mainResideProcess;  // a main process rank that a task maped to

    std::vector<ProcRank_t> m_TaskRankList;  //how task threads map to proc

    std::vector<ThreadId_t> m_LocalThreadList;

    std::map<ThreadId_t, ThreadRank_t> m_LocalThreadRegistry;

    std::map<ThreadRank_t, int> m_ThreadRank2Local;

    // output stream obj
    std::ofstream *m_procOstream;
    static boost::thread_specific_ptr<ThreadPrivateData> m_threadPrivateData;

    //
    void RegisterTask();

    // used for conduit to check all running threads of a task are finish,
    // and can destroy the conduit
    int m_activeLocalThreadCount;
    std::mutex m_activeLocalThreadMutex;
    std::condition_variable m_activeLocalThreadCond;

    // can't use taskbase obj directly
    TaskBase();


};

}

#endif

