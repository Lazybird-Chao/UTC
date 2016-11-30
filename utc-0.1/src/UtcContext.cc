#include "UtcContext.h"
#include "RootTask.h"
#include "ConduitManager.h"
#include "AffinityUtilities.h"
#include "gpu/UtcGpuBasics.h"
#if ENABLE_GPU_TASK
#include "gpu/CudaDeviceManager.h"
#endif

std::chrono::system_clock::time_point SYSTEM_START_TIME = std::chrono::system_clock::now();

namespace iUtc{

UtcBase* UtcContext::Utcbase_provider=0;
TaskId_t UtcContext::m_rootTaskId = -1;
int UtcContext::m_nCount = 0;
RootTask* UtcContext::root = nullptr;
int UtcContext::HARDCORES_TOTAL_CURRENT_NODE=getConcurrency();
int UtcContext::HARDCORES_ID_FOR_USING = 0;
UtcContext* UtcContext::m_ContextInstance = nullptr;

/* This singleton implementation is not thread safe.
 * However, the first time using it is at start of program,
 * usually happens only in main thread, so it's fine to use
 * this simple style.
 */
UtcContext& UtcContext::getContext(){
	if(m_ContextInstance == nullptr){
		m_ContextInstance = new UtcContext();
	}
	return *m_ContextInstance;
}

UtcContext& UtcContext::getContext(int &argc, char** &argv){
	if(m_ContextInstance==nullptr){
		m_ContextInstance = new UtcContext(argc, argv);
	}
	return *m_ContextInstance;
}

UtcContext::UtcContext()
{
    int argc=1;
    char *argv_storage[2];
    char **argv=argv_storage;
    argv[0]=(char*)"utc";
    argv[1]=NULL;
    initialize(argc, argv);
}

UtcContext::UtcContext(int& argc, char** &argv)
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

void UtcContext::Barrier()
{
	UtcContext::Utcbase_provider->Barrier();
}

////
TaskManager* UtcContext::getTaskManager()
{
    // notice the static
    static TaskManager* taskManager= TaskManager::getInstance();
    return taskManager;
}

//std::mutex* UtcContext::getCtxMutex(){
FastMutex* UtcContext::getCtxMutex(){
	return &m_ctxMutex;
}

SpinLock* UtcContext::getCtxSpinMutex(){
	return &m_ctxSpinMutex;
}


////
void UtcContext::initialize(int& argc, char** argv)
{
    // This initialize will only run once for the first time when creating Context obj.
    // TODO: change this class to "Singleton", even under multi thread.
   /* if(m_nCount && TaskManager::getCurrentTaskId()!= m_rootTaskId)
        return;*/
    if(m_nCount++!=0)
        return;

#ifdef USE_MPI_BASE
    Utcbase_provider = new UtcMpi::Utc(argc, argv);
#endif
    TaskManager* taskMgr= getTaskManager();    // The very first time and only this time to create a
                                               // TaskManager instance in current process.
    int nProcs= Utcbase_provider->numProcs();
    int pRank = Utcbase_provider->rank();

    //std::cout<<"In UtcContext.cc(l84): "<<nProcs<<std::endl;

    root= new RootTask(nProcs, pRank);          // The very first time and only this time to create a
                                                // RootTask instance in current process.

    m_rootTaskId= taskMgr->registerTask(root);

    taskMgr->setRootTask(root);

    ConduitManager* cdtMgr = ConduitManager::getInstance(); // The very first time and only this
    														// time to create a ConduitManager obj

#ifdef USE_MPI_BASE
    MPI_Barrier(*(root->getWorldComm()));
#endif

#ifdef USE_DEBUG_LOG
    std::ofstream *procOstream = root->getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"Utc context created on proc "<<pRank<<" ("<<getpid()<<")..."<<std::endl;
#endif

    /******** init gpu related staff *****/
#if ENABLE_GPU_TASK
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"Utc context init cudaDeviceManager!!!"<<std::endl;
#endif
    CudaDeviceManager& cudaDevMgr = CudaDeviceManager::getCudaDeviceManager();
	#if CUDA_CONTEXT_MAP_MODE==3
    	for(int i=0; i<cudaDevMgr.getNumDevices(); i++)
    		cudaDevMgr.initDevice(i);
	#endif
    	//std::cout<<cudaDevMgr.getNumDevices()<<std::endl;
#endif

}

void UtcContext::finalize()
{
#if ENABLE_GPU_TASK
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"Utc context destroy cudaDeviceManager!!!"<<std::endl;
#endif
    CudaDeviceManager& cudaDevMgr = CudaDeviceManager::getCudaDeviceManager();
	#if CUDA_CONTEXT_MAP_MODE==3
    	for(int i=0; i<cudaDevMgr.getNumDevices(); i++)
    		cudaDevMgr.resetDevice(i);
    	cudaDevMgr.~CudaDeviceManager();
	#endif
#endif
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

    // procOstream is new-ed in root's constructor, we delete it here as we still want
    // to use it before destruct the root obj
    std::ofstream *procOstream = root->getProcOstream();
    delete root;
    root = nullptr;
    if(Utcbase_provider)
    	delete Utcbase_provider;

    Utcbase_provider = nullptr;
#ifdef USE_DEBUG_LOG
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"Utc context destroyed!!!"<<std::endl;
#endif
    if(procOstream)
    {
    	if(procOstream->is_open())
    		procOstream->close();
    	delete procOstream;
    }
}

}// namespace iUtc
