#include "UtcContext.h"
#include "RootTask.h"
#include "ConduitManager.h"

#ifdef USE_DEBUG_LOG
	std::chrono::system_clock::time_point SYSTEM_START_TIME = std::chrono::system_clock::now();
#endif
namespace iUtc{

UtcBase* UtcContext::Utcbase_provider=0;
TaskId UtcContext::m_rootTaskId = 0;
int UtcContext::m_nCount = 0;
RootTask* UtcContext::root = nullptr;

UtcContext::UtcContext()
{
    int argc=1;
    char *argv_storage[2];
    char **argv=argv_storage;
    argv[0]=(char*)"utc";
    argv[1]=NULL;
    initialize(argc, argv);
}

UtcContext::UtcContext(int& argc, char**& argv)
{
    initialize(argc, argv);
}

UtcContext::~UtcContext()
{
    finalize();
}

int UtcContext::getProcRank()
{
    return UtcContext::Utcbase_provider->rank();
}

int UtcContext::numProcs()
{
    return UtcContext::Utcbase_provider->numProcs();
}

void UtcContext::getProcessorName(std::string& name)
{
    UtcContext::Utcbase_provider->getProcessorName(name);
}


////
TaskManager* UtcContext::getTaskManager()
{
    // notice the static
    static TaskManager* taskManager= TaskManager::getInstance();
    return taskManager;
}


////
void UtcContext::initialize(int& argc, char** argv)
{
    // This initialize will only run once for the first time when creating Context obj.
    // TODO: change this class to "Singleton", even under multi thread.
    if(m_nCount && TaskManager::getCurrentTaskId()!= m_rootTaskId)
        return;
    if(m_nCount++!=0)
        return;

#ifdef USE_MPI_BASE
    Utcbase_provider = new UtcMpi::Utc(argc, argv);
#endif
    TaskManager* taskMgr= getTaskManager();    // The very first time and only this time to create a
                                               // TaskManager instance in current process.
    int nProcs= Utcbase_provider->numProcs();
    int pRank = Utcbase_provider->rank();

    root= new RootTask(nProcs, pRank);          // The very first time and only this time to create a
                                                // RootTask instance in current process.

    m_rootTaskId= taskMgr->registerTask(root);

    taskMgr->setRootTask(root);

    ConduitManager* cdtMgr = ConduitManager::getInstance(); // The very first time and only this
    														// time to create a ConduitManager obj
}

void UtcContext::finalize()
{
    if(TaskManager::getCurrentTaskId()!= m_rootTaskId)
        return;
    if(--m_nCount != 0)
        return;

    // we use static local var to creat singleton for cdtMgr, so system will
    // destroy it automatically, we should not delete it here.
    /*ConduitManager* cdtMgr = ConduitManager::getInstance();
    delete cdtMgr;*/

    // we explicitly destroy taskMgr singleton here
    // as taskMgr singleton is create by new(), it need be destryed by delete()
    // we didn't use define static obj member inside singleton to automatically
    // do this delete, we do it here manually
    TaskManager* taskMgr= getTaskManager();
    taskMgr->unregisterTask(root, 0);
    delete taskMgr;

    delete root;
    root = nullptr;

    delete Utcbase_provider;
    Utcbase_provider = nullptr;
}

}// namespace iUtc
