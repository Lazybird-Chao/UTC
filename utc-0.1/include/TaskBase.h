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
    std::vector<ProcRank> getTaskMapList();

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

    int    m_numLocalThreads;

    int    m_numTotalThreads;

    ProcRank m_processRank;

    std::vector<ProcRank> m_TaskMapList;

    std::vector<ThreadId> m_LocalThreadList;

    std::map<ThreadId, ThreadRank> m_LocalThreadRegistry;

    //
    TaskId RegisterTask();

    TaskBase();


};

}

#endif

