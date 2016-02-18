#include "InprocConduit.h"
#include "UtcBasics.h"
#include "../include/TaskUtilities.h"
#include "LockFreeRingbufferQueue.h"

#include <cstdlib>
#include <cstring>
#include <chrono>



namespace iUtc{

thread_local std::ofstream *InprocConduit::m_threadOstream = nullptr;


/*
 *
 */
InprocConduit::InprocConduit(TaskBase *srctask, TaskBase *dsttask, int cdtId)
:ConduitBase(){
	m_conduitId = cdtId;
    m_srcTask = srctask;
    m_dstTask = dsttask;
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
	m_srcBuffQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
		m_numSrcLocalThreads, m_numDstLocalThreads);
    m_srcInnerMsgQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
        m_numSrcLocalThreads, m_numDstLocalThreads);
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = new MsgInfo_t;
        tmpbuff->smallDataBuff = (void*)malloc(CONDUIT_BUFFER_SIZE);
        m_srcInnerMsgQueue->push(tmpbuff);
    }
    m_srcBuffMap.clear();
	
    m_srcOpTokenFlag = new int[m_numSrcLocalThreads];
    for(int i=0; i<m_numSrcLocalThreads; i++){
        m_srcOpTokenFlag[i]=0;
        boost::latch *tmp_latch = new boost::latch(1);
        m_srcOpThreadLatch.push_back(tmp_latch);       
    }

    m_srcUsingPtrFinishFlag = new std::atomic<int>[m_numSrcLocalThreads];
    for(int i=0; i<m_numSrcLocalThreads; i++){
        m_srcUsingPtrFinishFlag[i] = 0;
    }


    // init dst side
	m_dstBuffQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
		m_numDstLocalThreads, m_numSrcLocalThreads);
    m_dstInnerMsgQueue = new LockFreeQueue<MsgInfo_t, INPROC_CONDUIT_CAPACITY_DEFAULT>(
        m_numSrcLocalThreads, m_numDstLocalThreads);
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = new MsgInfo_t;
        tmpbuff->smallDataBuff = (void*)malloc(CONDUIT_BUFFER_SIZE);
        m_dstInnerMsgQueue->push(tmpbuff);
    }
    m_dstBuffMap.clear();

    m_dstOpTokenFlag = new int[m_numDstLocalThreads];
    for(int i=0; i<m_numDstLocalThreads; i++){
        m_dstOpTokenFlag[i]=0;
        boost::latch *tmp_latch = new boost::latch(1);
        m_dstOpThreadLatch.push_back(tmp_latch);
    }

    m_dstUsingPtrFinishFlag = new std::atomic<int>[m_numDstLocalThreads];
    for(int i=0; i<m_numDstLocalThreads; i++){
        m_dstUsingPtrFinishFlag[i] = 0;
    }




    // init readby, writeby related
    m_readbyFinishSet.clear();
    m_writebyFinishSet.clear();


    // init Async op related
    m_srcAsyncReadFinishSet.clear();
    m_srcAsyncWriteFinishSet.clear();
    m_srcNewAsyncWork=false;
    m_srcAsyncWorkerCloseSig=false;
    m_srcAsyncWorkerOn=false;
    m_srcAsyncWorkQueue.clear();

    m_dstAsyncReadFinishSet.clear();
    m_dstAsyncWriteFinishSet.clear();
    m_dstNewAsyncWork=false;
    m_dstAsyncWorkerCloseSig=false;
    m_dstAsyncWorkerOn=false;
    m_dstAsyncWorkQueue.clear();


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
    *procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
#endif

}



void InprocConduit::clear(){

    //
    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = m_srcInnerMsgQueue->pop();
        free(tmpbuff->smallDataBuff);
    }
    m_srcBuffMap.clear();
    delete m_srcOpTokenFlag;
    delete m_srcUsingPtrFinishFlag;
    for(int i=0; i<m_numSrcLocalThreads; i++){
        delete m_srcOpThreadLatch[i];       
    }
    m_srcOpThreadLatch.clear();

    for(int i=0; i< INPROC_CONDUIT_CAPACITY_DEFAULT; i++)
    {
        MsgInfo_t *tmpbuff = m_dstInnerMsgQueue->pop();
        free(tmpbuff->smallDataBuff);
    }
    m_dstBuffMap.clear();
    delete m_dstOpTokenFlag;
    delete m_dstUsingPtrFinishFlag;
    for(int i=0; i<m_numDstLocalThreads; i++){
        delete m_dstOpThreadLatch[i];       
    }
    m_dstOpThreadLatch.clear();

    //
    m_readbyFinishSet.clear();
    m_writebyFinishSet.clear();

    //
    m_srcAsyncReadFinishSet.clear();
    m_srcAsyncWriteFinishSet.clear();
    m_srcAsyncWorkQueue.clear();

    m_dstAsyncReadFinishSet.clear();
    m_dstAsyncWriteFinishSet.clear();
    m_dstAsyncWorkQueue.clear();
}


} // end namespace iUtc
