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

    int toLocal(ThreadRank trank);

    //
    int getNumLocalThreads();
    std::vector<ThreadId> getLocalThreadList();

    int getNumTotalThreads();

    ProcRank getCurrentProcRank();

    bool isLocal(ThreadRank tRank);

    ThreadRank getThreadRankById(ThreadId tid);

    static ThreadPrivateData* getThreadPrivateData();
    static void setThreadPrivateData(ThreadPrivateData* tpd);

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

    // can't use taskbase obj directly
    TaskBase();


};

}

#endif

