#ifndef UTC_TASKBASE_H_
#define UTC_TASKBASE_H_

#include "RankList.h"
#include "UtcBasics.h"

#include <map>
#include <vector>
#include <string>

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


    //
    int getNumLocalThreads();
    std::vector<ThreadId> getLocalThreadList();

    int getNumTotalThreads();

    ProcRank getCurrentProcRank();

    bool isLocal(ThreadRank tRank);

    ThreadRank getThreadRankById(ThreadId tid);
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

    std::vector<ProcRank> m_TaskRankList;

    std::vector<ThreadId> m_LocalThreadList;

    std::map<ThreadId, ThreadRank> m_LocalThreadRegistry;

    //
    TaskId RegisterTask();

    // can't use taskbase obj directly
    TaskBase();


};

}

#endif

