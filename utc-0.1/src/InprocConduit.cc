#include "InprocConduit.h"
#include "UtcBasics.h"
#include "Task_Utilities.h"

#include <map>
#include <cstdlib>
#include <cstring>
#include <chrono>


namespace iUtc
{

thread_local std::ofstream *InprocConduit::m_threadOstream = nullptr;



InprocConduit::InprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId, int capacity)
:ConduitBase()
{

	m_conduitId = cdtId;
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();
    m_capacity = capacity;
    m_noFinishedOpCapacity = NO_FINISHED_OP_MAX;

    initInprocConduit();

}


void InprocConduit::initInprocConduit()
{
	/*if(m_srcTask->isActiveOnCurrentProcess()== false && m_dstTask->isActiveOnCurrentProcess()== false)
	{	// no thread for src and dst running on this process, no need to init the conduit obj
		return;
	}
	m_Name = m_srcTask->getName()+"<=>"+m_dstTask->getName();
	m_conduitId = ConduitManager::getNewConduitId();*/


	m_srcAvailableBuffCount = m_capacity;
	m_srcAvailableBuff = new BuffInfo[m_capacity];
	for(int i=0; i<m_capacity; i++)
	{
	    m_srcAvailableBuff[i].buffSize = CONDUIT_BUFFER_SIZE;
	    m_srcAvailableBuff[i].bufferPtr = malloc(CONDUIT_BUFFER_SIZE);
	}
	m_srcBuffPool.clear();
	m_srcBuffPoolWaitlist.clear();
	m_srcBuffIdx.clear();
	std::vector<std::mutex> *tmp1_mutexlist = new std::vector<std::mutex>(m_capacity);
	m_srcBuffAccessMutex.swap(*tmp1_mutexlist);
	std::vector<std::condition_variable> *tmp1_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_srcBuffDataWrittenCond.swap(*tmp1_condlist);
	std::vector<std::condition_variable> *tmp2_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_srcBuffDataReadCond.swap(*tmp2_condlist);
	m_srcBuffDataWrittenFlag.clear();
	m_srcBuffDataReadFlag.clear();
	for(int i = 0; i< m_capacity; i++)
	{
		m_srcBuffIdx.push_back(i);
		m_srcBuffDataWrittenFlag.push_back(0);
		m_srcBuffDataReadFlag.push_back(0);
	}

	m_srcAvailableNoFinishedOpCount = m_noFinishedOpCapacity;
	m_srcOpRotateCounter = new int[m_noFinishedOpCapacity+1];
	m_srcOpRotateFinishFlag = new int[m_noFinishedOpCapacity+1];
	for(int i =0; i<m_noFinishedOpCapacity; i++)
	{
	    m_srcOpRotateCounter[i]=0;
	    m_srcOpRotateCounter[i]=0;
	}
	m_srcOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
    for(int i =0; i<m_srcTask->getNumTotalThreads();i++)
    {
        m_srcOpRotateCounterIdx[i] = 0;
    }





	m_dstAvailableBuffCount = m_capacity;
	m_dstAvailableBuff = new BuffInfo[m_capacity];
    for(int i=0; i<m_capacity; i++)
    {
        m_dstAvailableBuff[i].buffSize = CONDUIT_BUFFER_SIZE;
        m_dstAvailableBuff[i].bufferPtr = malloc(CONDUIT_BUFFER_SIZE);
    }
	m_dstBuffPool.clear();
	m_dstBuffPoolWaitlist.clear();
	m_dstBuffIdx.clear();
	//m_dstBuffAccessMutex.clear();
	std::vector<std::mutex> *tmp2_mutexlist= new std::vector<std::mutex>(m_capacity);
	m_dstBuffAccessMutex.swap(*tmp2_mutexlist);
	std::vector<std::condition_variable> *tmp3_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_dstBuffDataWrittenCond.swap(*tmp3_condlist);
	std::vector<std::condition_variable> *tmp4_condlist = new std::vector<std::condition_variable>(m_capacity);
	m_dstBuffDataReadCond.swap(*tmp4_condlist);
	m_dstBuffDataWrittenFlag.clear();
	m_dstBuffDataReadFlag.clear();
	for(int i = 0; i< m_capacity; i++)
	{
		m_dstBuffIdx.push_back(i);
		m_dstBuffDataWrittenFlag.push_back(0);
		m_dstBuffDataReadFlag.push_back(0);
	}

	m_dstAvailableNoFinishedOpCount = m_noFinishedOpCapacity;
    m_dstOpRotateCounter = new int[m_noFinishedOpCapacity+1];
    m_dstOpRotateFinishFlag = new int[m_noFinishedOpCapacity+1];
    for(int i =0; i<m_noFinishedOpCapacity; i++)
    {
        m_dstOpRotateCounter[i]=0;
        m_dstOpRotateCounter[i]=0;
    }
    m_dstOpRotateCounterIdx = new int[m_dstTask->getNumTotalThreads()];
    for(int i =0; i<m_dstTask->getNumTotalThreads();i++)
    {
        m_dstOpRotateCounterIdx[i] = 0;
    }


	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();



#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] initiated..."<<std::endl;
#endif

}


InprocConduit::~InprocConduit()
{

	clear();

#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    *procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
#endif

}

void InprocConduit::clear()
{

	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();

    m_srcBuffIdx.clear();
    m_srcBuffAccessMutex.clear();
    m_srcBuffDataWrittenCond.clear();
    m_srcBuffDataReadCond.clear();
    m_srcBuffDataWrittenFlag.clear();
    m_srcBuffDataReadFlag.clear();
    delete m_srcOpRotateCounter;
    delete m_srcOpRotateCounterIdx;
    delete m_srcOpRotateFinishFlag;


    m_dstBuffIdx.clear();
    m_dstBuffAccessMutex.clear();
    m_dstBuffDataWrittenCond.clear();
    m_dstBuffDataReadCond.clear();
    m_dstBuffDataWrittenFlag.clear();
    m_dstBuffDataReadFlag.clear();
    delete m_dstOpRotateCounter;
    delete m_dstOpRotateCounterIdx;
    delete m_dstOpRotateFinishFlag;

    m_srcBuffPool.clear();
    m_srcBuffPoolWaitlist.clear();
    for(int i=0; i<m_capacity; i++)
    {
        free(m_srcAvailableBuff[i].bufferPtr);
    }
    delete m_srcAvailableBuff;


    m_dstBuffPool.clear();
    m_dstBuffPoolWaitlist.clear();
    for(int i=0; i<m_capacity; i++)
    {
        free(m_dstAvailableBuff[i].bufferPtr);
    }
    delete m_dstAvailableBuff;


}



}// namespace iUtc
