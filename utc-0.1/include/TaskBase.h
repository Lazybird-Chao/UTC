#ifndef UTC_TASKBASE_H_
#define UTC_TASKBASE_H_

#include "RankList.h"
#include "UtcBasics.h"
#include "TaskInfo.h"


#include <map>
#include <vector>
#include <string>
#include <boost/thread/tss.hpp>
#include <boost/thread/thread.hpp>
#include <mutex>
#include <condition_variable>

namespace iUtc{

class TaskBase{
public:
    std::string getName();

    TaskId getTaskId();

    TaskBase* getParentTask();
    TaskId getParentTaskId();

    //
    std::vector<ProcRank> getTaskRankList();

    int getNumProcesses();

    // convert thread global rank(among all processes of a task) to local rank
    // (the rank on current process)
    int toLocal(ThreadRank trank);

    //
    int getNumLocalThreads();
    std::vector<ThreadId> getLocalThreadList();

    int getNumTotalThreads();

    ProcRank getCurrentProcRank();

    // return if thread with rank value 'tRank' is on current process
    bool isLocal(ThreadRank tRank);

    ThreadRank getThreadRankById(ThreadId tid);

    static ThreadPrivateData* getThreadPrivateData();
    static void setThreadPrivateData(ThreadPrivateData* tpd);

    bool isActiveOnCurrentProcess();

    bool hasActiveLocalThread();
    void waitLocalThreadFinish();

    //
    virtual ~TaskBase();

protected:
    std::string m_Name;

    TaskId m_TaskId;

    TaskId m_ParentTaskId;

    int    m_numProcesses;

    // process related, for a task obj, in different process it may get
    // different value, depends on the ranklist of the task
    int    m_numLocalThreads;

    int    m_numTotalThreads;

    ProcRank m_processRank;    // the current running process's rank value

    std::vector<ProcRank> m_TaskRankList;  //how task threads map to proc

    std::vector<ThreadId> m_LocalThreadList;

    std::map<ThreadId, ThreadRank> m_LocalThreadRegistry;

    std::map<ThreadRank, int> m_ThreadRank2Local;

    // output stream obj
    std::ofstream *m_procOstream;
    static boost::thread_specific_ptr<ThreadPrivateData> m_threadPrivateData;

    //
    void RegisterTask();

    // used for conduit to check all reated task thread finish,
    // and can destroy the conduit
    int m_activeLocalThreadCount;
    std::mutex m_activeLocalThreadMutex;
    std::condition_variable m_activeLocalThreadCond;

    // can't use taskbase obj directly
    TaskBase();


};

}

#endif

