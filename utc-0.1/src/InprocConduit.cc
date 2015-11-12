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

/*InprocConduit::InprocConduit()
:ConduitBase()
{
    m_srcTask = nullptr;
    m_dstTask = nullptr;
    m_srcId = -1;
    m_dstId = -1;

    m_numSrcLocalThreads = 0;
    m_numDstLocalThreads = 0;
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
#ifdef USE_DEBUG_LOG
    std::ofstream *procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: [dummy conduit] constructed..."<<std::endl;
#endif
    //initConduit();   call init through connect()

}
void InprocConduit::checkOnSameProc(TaskBase* src, TaskBase* dst)
{
	if((src->isActiveOnCurrentProcess()== false && dst->isActiveOnCurrentProcess()== true) ||
			(src->isActiveOnCurrentProcess()== true && dst->isActiveOnCurrentProcess()==false))
	{
		std::cout<<"Error, two Tasks are not running on same process!"<<std::endl;
		exit(1);
	}
}
InprocConduit::InprocConduit(TaskBase* srctask, TaskBase* dsttask)
:ConduitBase()
{
	checkOnSameProc(srctask, dsttask);
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();
    m_capacity = CONDUIT_CAPACITY_DEFAULT;
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] constructed..."<<std::endl;
#endif
    initInprocConduit();

}*/

InprocConduit::InprocConduit(TaskBase* srctask, TaskBase* dsttask, int cdtId, int capacity)
:ConduitBase()
{
	/*checkOnSameProc(srctask,dsttask);*/
	m_conduitId = cdtId;
    m_srcTask = srctask;
    m_dstTask = dsttask;
    m_srcId = m_srcTask->getTaskId();
    m_dstId = m_dstTask->getTaskId();

    m_numSrcLocalThreads = srctask->getNumLocalThreads();
    m_numDstLocalThreads = dsttask->getNumLocalThreads();
    m_capacity = capacity;
    /*if(capacity > CONDUIT_CAPACITY_MAX)
    {
    	m_capacity = CONDUIT_CAPACITY_MAX;
    }
    else
    	m_capacity = capacity;
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] constructed..."<<std::endl;
#endif*/

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
	m_srcBuffPool.clear();
	m_srcBuffPoolWaitlist.clear();
	m_srcBuffIdx.clear();
	//m_srcBuffAccessMutex.clear();
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
	m_srcWriteOpRotateCounter = new int[m_capacity+1];
	m_srcReadOpRotateCounter = new int[m_capacity +1];
	m_srcWriteOpRotateFinishFlag = new int[m_capacity+1];
	m_srcReadOpRotateFinishFlag = new int[m_capacity+1];
	for(int i =0;i<m_capacity+1; i++)
	{
		m_srcWriteOpRotateCounter[i]=0;
		m_srcReadOpRotateCounter[i]=0;
		m_srcWriteOpRotateFinishFlag[i]=0;
		m_srcReadOpRotateFinishFlag[i]=0;
	}
	m_srcWriteOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	//actually only use the local threads in one process, other thread pos is not used
	m_srcReadOpRotateCounterIdx = new int[m_srcTask->getNumTotalThreads()];
	for(int i =0; i<m_srcTask->getNumTotalThreads();i++)
	{
		m_srcWriteOpRotateCounterIdx[i] = 0;
		m_srcReadOpRotateCounterIdx[i]=0;
	}


	m_dstAvailableBuffCount = m_capacity;
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
	m_dstWriteOpRotateCounter = new int[m_capacity+1];
	m_dstReadOpRotateCounter = new int[m_capacity +1];
	m_dstWriteOpRotateFinishFlag = new int[m_capacity+1];
	m_dstReadOpRotateFinishFlag = new int[m_capacity+1];
	for(int i =0;i<m_capacity+1; i++)
	{
		m_dstWriteOpRotateCounter[i]=0;
		m_dstReadOpRotateCounter[i]=0;
		m_dstWriteOpRotateFinishFlag[i]=0;
		m_dstReadOpRotateFinishFlag[i]=0;
	}
	m_dstWriteOpRotateCounterIdx = new int[m_dstTask->getNumTotalThreads()];
	m_dstReadOpRotateCounterIdx = new int[m_dstTask->getNumTotalThreads()];
	for(int i =0; i<m_dstTask->getNumTotalThreads();i++)
	{
		m_dstWriteOpRotateCounterIdx[i] = 0;
		m_dstReadOpRotateCounterIdx[i]=0;
	}


	m_readbyFinishSet.clear();
	m_writebyFinishSet.clear();


	/*ConduitManager* cdtMgr = ConduitManager::getInstance();
	m_cdtMgr = cdtMgr;
	cdtMgr->registerConduit(this, m_conduitId);*/


#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] initiated..."<<std::endl;
#endif

}


InprocConduit::~InprocConduit()
{
    /*if(m_srcTask)
    {
        if(TaskManager::hasTaskItem(m_srcId))
        {   // task not destroyed
            if(m_srcTask->hasActiveLocalThread())
            {
                // there are task threads still running
                m_srcTask->waitLocalThreadFinish();
            }
        }
    }
    if(m_dstTask)
    {
        if(TaskManager::hasTaskItem(m_dstId))
            if(m_dstTask->hasActiveLocalThread())
                m_dstTask->waitLocalThreadFinish();
    }
    // delete this conduit item from conduit registry
	m_cdtMgr->unregisterConduit(this, m_conduitId);*/

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
    delete m_srcWriteOpRotateCounter;
    delete m_srcReadOpRotateCounter;
    delete m_srcWriteOpRotateCounterIdx;
    delete m_srcReadOpRotateCounterIdx;
    delete m_srcWriteOpRotateFinishFlag;
    delete m_srcReadOpRotateFinishFlag;

    m_dstBuffIdx.clear();
    m_dstBuffAccessMutex.clear();
    m_dstBuffDataWrittenCond.clear();
    m_dstBuffDataReadCond.clear();
    m_dstBuffDataWrittenFlag.clear();
    m_dstBuffDataReadFlag.clear();
    delete m_dstWriteOpRotateCounter;
    delete m_dstReadOpRotateCounter;
    delete m_dstWriteOpRotateCounterIdx;
    delete m_dstReadOpRotateCounterIdx;
    delete m_dstWriteOpRotateFinishFlag;
    delete m_dstReadOpRotateFinishFlag;

    for(std::map<MessageTag, BuffInfo*>::iterator it = m_srcBuffPool.begin();
            it != m_srcBuffPool.end(); ++it)
    {
        if((it->second)->dataPtr)
            free((it->second)->dataPtr);
        delete it->second;
    }
    m_srcBuffPool.clear();
    m_srcBuffPoolWaitlist.clear();

    for(std::map<MessageTag, BuffInfo*>::iterator it = m_dstBuffPool.begin();
            it != m_dstBuffPool.end(); ++it)
    {
        if((it->second)->dataPtr)
            free((it->second)->dataPtr);
        delete it->second;
    }
    m_dstBuffPool.clear();
    m_dstBuffPoolWaitlist.clear();

/*
#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    if(m_srcTask && m_dstTask)
    {
        *procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
                <<"] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
    }
    else
    {
        *procOstream<<"Conduit: [dummy conduit] destroyed on proc "<<m_srcTask->getCurrentProcRank()<<"!!!"<<std::endl;
    }
#endif
*/
}

// can't change capacity after conduit is created
/*void Conduit::setCapacity(int capacity)
{
    m_capacity = capacity;
}*/

/*int InprocConduit::getCapacity()
{
    return m_capacity;
}

std::string InprocConduit::getName()
{
	return m_Name;
}

TaskBase* InprocConduit::getSrcTask()
{
    return m_srcTask;
}

TaskBase* InprocConduit::getDstTask()
{
    return m_dstTask;
}

TaskBase* InprocConduit::getAnotherTask()
{
	static thread_local int myTaskid = -1;
	if(myTaskid == -1)
	{
		myTaskid = TaskManager::getCurrentTaskId();
	}
	if(myTaskid == m_srcId)
		return m_dstTask;
	else if(myTaskid == m_dstId)
		return m_srcTask;
	else
	{
		std::cout<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

}

void InprocConduit::Connect(TaskBase* src, TaskBase* dst)
{
    if(m_srcTask || m_dstTask)
    {
        std::cout<<"Error, already connected to some Task"<<std::endl;
        exit(1);
    }
    checkOnSameProc(src, dst);
    m_srcTask = src;
    m_dstTask = dst;
    m_srcId = src->getTaskId();
    m_dstId = dst->getTaskId();
    m_numSrcLocalThreads = src->getNumLocalThreads();
    m_numDstLocalThreads = dst->getNumLocalThreads();
#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
				<<"] connected..."<<std::endl;
#endif
    initInprocConduit();
    return;
}

ConduitId InprocConduit::getConduitId()
{
    return m_conduitId;
}*/




}// namespace iUtc
