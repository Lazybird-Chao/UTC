#include "InprocConduit.h"
#include "UtcBasics.h"
#include "../include/TaskUtilities.h"
#include "LockFreeRingbufferQueue.h"

#include <cstdlib>
#include <cstring>
#include <chrono>

struct timespec SHORT_PERIOD{0, 1000}; // 1us
struct timespec LONG_PERIOD{0,100000}; // 100us

namespace iUtc{

thread_local std::ofstream *InprocConduit::m_threadOstream = nullptr;


/*
 *
 */
InprocConduit::InprocConduit(TaskBase *srctask, TaskBase *dsttask, int cdtId, std::string name)
:ConduitBase(){
	m_conduitId = cdtId;
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_Name = name;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();

    initInprocConduit();
}

/*
 *
 */
void InprocConduit::initInprocConduit(){

    // init src side
	// src push, dst pop, extra one is for async op thread
	m_srcBuffQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
		m_numSrcLocalThreads+1, m_numDstLocalThreads+1);
	// src pop, dst push, extra one is for async op thread
    m_srcInnerMsgQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
        m_numDstLocalThreads+1, m_numSrcLocalThreads+1);

    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = new MsgInfo_t;
        tmpbuff->smallDataBuff = (void*)malloc(CONDUIT_BUFFER_SIZE);
        m_srcInnerMsgQueue->initPush(tmpbuff);
    }
    m_srcBuffMap.clear();

    m_srcOpTokenFlag = new int[m_numSrcLocalThreads];
    /*m_srcOpThreadAtomic = new std::atomic<int>[m_numSrcLocalThreads];
    for(int i=0; i<m_numSrcLocalThreads; i++){
        m_srcOpTokenFlag[i]=0;
        boost::latch *tmp_latch = new boost::latch(1);
        m_srcOpThreadLatch.push_back(tmp_latch);
        m_srcOpThreadAtomic[i].store(1);
    }*/
    for(int i=0; i<m_numSrcLocalThreads; i++)
    	m_srcOpTokenFlag[i]=0;
    m_srcOpThreadAvailable.push_back(new std::atomic<int>(0));
    m_srcOpThreadFinish.push_back(new std::atomic<intptr_t>(new boost::latch(1)));

    // extra one for async op
    m_srcUsingPtrFinishFlag = new std::atomic<int>[m_numSrcLocalThreads+1];
    for(int i=0; i<m_numSrcLocalThreads+1; i++){
        m_srcUsingPtrFinishFlag[i].store(0);
    }



    // init dst side
    // dst push, src pop, extra one is for async op thread
	m_dstBuffQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
		m_numDstLocalThreads+1, m_numSrcLocalThreads+1);
	// dst pop, src push
    m_dstInnerMsgQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
    		m_numSrcLocalThreads+1, m_numDstLocalThreads+1);
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = new MsgInfo_t;
        tmpbuff->smallDataBuff = (void*)malloc(CONDUIT_BUFFER_SIZE);
        m_dstInnerMsgQueue->initPush(tmpbuff);
    }
    m_dstBuffMap.clear();

    m_dstOpTokenFlag = new int[m_numDstLocalThreads];
    /*m_dstOpThreadAtomic = new std::atomic<int>[m_numDstLocalThreads];
    for(int i=0; i<m_numDstLocalThreads; i++){
        m_dstOpTokenFlag[i]=0;
        boost::latch *tmp_latch = new boost::latch(1);
        m_dstOpThreadLatch.push_back(tmp_latch);
        m_dstOpThreadAtomic[i].store(1);
    }*/
    for(int i=0; i<m_numDstLocalThreads; i++)
		m_dstOpTokenFlag[i]=0;
	m_dstOpThreadAvailable.push_back(new std::atomic<int>(0));
	m_dstOpThreadFinish.push_back(new std::atomic<intptr_t>(new boost::latch(1)));

    // extra one for async op
    m_dstUsingPtrFinishFlag = new std::atomic<int>[m_numDstLocalThreads+1];
    for(int i=0; i<m_numDstLocalThreads+1; i++){
        m_dstUsingPtrFinishFlag[i].store(0);
    }



#ifdef ENABLE_OPBY_FINISH
    // init readby, writeby related
    m_readbyFinishSet.clear();
    m_writebyFinishSet.clear();
#endif


    // init Async op related
    m_srcAsyncReadFinishSet.clear();
    m_srcAsyncWriteFinishSet.clear();
    m_dstAsyncReadFinishSet.clear();
    m_dstAsyncWriteFinishSet.clear();
    m_srcAsyncWorkQueue = new LockFreeQueue<AsyncWorkArgs_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
            m_numSrcLocalThreads, 1);
    m_dstAsyncWorkQueue = new LockFreeQueue<AsyncWorkArgs_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
            m_numDstLocalThreads, 1);
    m_srcAsyncOpTokenFlag = new int[m_numSrcLocalThreads];
    /*m_srcAsyncOpThreadAtomic = new std::atomic<int>[m_numSrcLocalThreads];
    for(int i=0;i< m_numSrcLocalThreads; i++){
    	m_srcAsyncOpTokenFlag[i] = 0;
    	m_srcAsyncOpThreadAtomic[i].store(1);
    }*/
    m_dstAsyncOpTokenFlag = new int[m_numDstLocalThreads];
    /*m_dstAsyncOpThreadAtomic = new std::atomic<int>[m_numDstLocalThreads];
    for(int i=0; i<m_numDstLocalThreads; i++){
    	m_dstAsyncOpTokenFlag[i] = 0;
    	m_dstAsyncOpThreadAtomic[i].store(1);
    }*/
    for(int i=0; i<m_numSrcLocalThreads; i++)
		m_srcAsyncOpTokenFlag[i]=0;
	m_srcAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
	m_srcAsyncOpThreadFinish.push_back(new std::atomic<int>(1));
    for(int i=0; i<m_numDstLocalThreads; i++)
		m_dstAsyncOpTokenFlag[i]=0;
	m_dstAsyncOpThreadAvailable.push_back(new std::atomic<int>(0));
	m_dstAsyncOpThreadFinish.push_back(new std::atomic<int>(1));

    m_srcAsyncWorkerOn.store(false);
    m_dstAsyncWorkerOn.store(false);



#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] initiated..."<<std::endl;
#endif

}


InprocConduit::~InprocConduit(){

    clear();

#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"InprocAsyncWokerCount: "<<m_asyncWorkerCount<<std::endl;
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"InprocConduit: ["<<m_Name<<"] destroyed!!!"<<std::endl;
#endif

}



void InprocConduit::clear(){

    //
	m_srcInnerMsgQueue->setupEndPop();
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = m_srcInnerMsgQueue->endPop();

        free(tmpbuff->smallDataBuff);
    }
    m_srcBuffMap.clear();
    delete m_srcOpTokenFlag;
    /*delete m_srcOpThreadAtomic;*/
    delete m_srcUsingPtrFinishFlag;
/*
    for(int i=0; i<m_numSrcLocalThreads; i++){
        delete m_srcOpThreadLatch[i];       
    }
    m_srcOpThreadLatch.clear();
*/
    m_srcOpThreadAvailable.clear();
    m_srcOpThreadFinish.clear();
    m_dstOpThreadAvailable.clear();
    m_dstOpThreadFinish.clear();

    //
    m_dstInnerMsgQueue->setupEndPop();
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = m_dstInnerMsgQueue->endPop();
        free(tmpbuff->smallDataBuff);
    }
    m_dstBuffMap.clear();
    delete m_dstOpTokenFlag;
    /*delete m_dstOpThreadAtomic;*/
    delete m_dstUsingPtrFinishFlag;
/*    for(int i=0; i<m_numDstLocalThreads; i++){
        delete m_dstOpThreadLatch[i];       
    }
    m_dstOpThreadLatch.clear();
*/

    //
#ifdef ENABLE_OPBY_FINISH
    m_readbyFinishSet.clear();
    m_writebyFinishSet.clear();
#endif

    //
    // check if there's async worker thread running
    while(m_srcAsyncWorkerOn.load()==true || m_dstAsyncWorkerOn.load()==true){
    	usleep(1);
    }
    m_srcAsyncReadFinishSet.clear();
    m_srcAsyncWriteFinishSet.clear();
    delete m_srcAsyncWorkQueue;
    delete m_srcAsyncOpTokenFlag;
    /*delete m_srcAsyncOpThreadAtomic;*/
    m_srcAsyncOpThreadAvailable.clear();
    m_srcAsyncOpThreadFinish.clear();
    m_dstAsyncOpThreadAvailable.clear();
    m_dstAsyncOpThreadFinish.clear();

    m_dstAsyncReadFinishSet.clear();
    m_dstAsyncWriteFinishSet.clear();
    delete m_dstAsyncWorkQueue;
	delete m_dstAsyncOpTokenFlag;
	/*delete m_dstAsyncOpThreadAtomic;*/


}


} // end namespace iUtc
