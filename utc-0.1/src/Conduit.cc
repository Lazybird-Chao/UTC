#include "Conduit.h"
#include "../include/TaskUtilities.h"
#include "UtcBasics.h"
#include "ConduitManager.h"
#include "InprocConduit.h"
#include "XprocConduit.h"
#include "TaskManager.h"

namespace iUtc{

Conduit::Conduit()
{
	m_srcTask = nullptr;
	m_dstTask = nullptr;
	m_srcId = -1;
	m_dstId = -1;

	m_capacity = INPROC_CONDUIT_CAPACITY_DEFAULT;
	m_Name = "";
	m_conduitId = ConduitManager::getNewConduitId();
	m_cdtMgr = ConduitManager::getInstance();
	m_cdtMgr->registerConduit(this, m_conduitId);

	m_realConduitPtr = nullptr;

#ifdef USE_DEBUG_LOG
	std::ofstream *procOstream = getProcOstream();
	PRINT_TIME_NOW(*procOstream)
	*procOstream<<"Conduit: [dummy conduit] constructed..."<<std::endl;
#endif
}

Conduit::Conduit(TaskBase* srctask, TaskBase* dsttask)
{
	initConduit(srctask, dsttask, INPROC_CONDUIT_CAPACITY_DEFAULT);
}


int Conduit::initConduit(TaskBase* srctask, TaskBase* dsttask, int capacity){
	m_srcTask = srctask;
	m_dstTask = dsttask;
	m_srcId = m_srcTask->getTaskId();
	m_dstId = m_dstTask->getTaskId();

	if(capacity > INPROC_CONDUIT_CAPACITY_MAX)
	{
		m_capacity = INPROC_CONDUIT_CAPACITY_MAX;
	}
	else
	{
		m_capacity = capacity;
	}
	m_Name = m_srcTask->getName()+"<=>"+m_dstTask->getName();
	m_conduitId = ConduitManager::getNewConduitId();
	m_cdtMgr = ConduitManager::getInstance();
	m_cdtMgr->registerConduit(this, m_conduitId);

#ifdef USE_DEBUG_LOG
	std::ofstream* procOstream = getProcOstream();
#endif

	/*
	 * we just now only allow simple conduit between two tasks.
	 * This requires that each task only active on one process.
	 */
	if(srctask->getNumProcesses() == 1 && dsttask->getNumProcesses() == 1){
	 	if(srctask->isActiveOnCurrentProcess()==true &&
				dsttask->isActiveOnCurrentProcess()==true)
		{
			m_realConduitPtr = new InprocConduit(srctask, dsttask, m_conduitId, m_Name);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*procOstream)
		*procOstream<<"InprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
						<<"] constructed..."<<std::endl;
#endif
		}
		else if((srctask->isActiveOnCurrentProcess()== false && dsttask->isActiveOnCurrentProcess()== true) ||
				(srctask->isActiveOnCurrentProcess()== true && dsttask->isActiveOnCurrentProcess()==false))
		{
			m_realConduitPtr = new XprocConduit(srctask, dsttask, m_conduitId, m_Name);
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*procOstream)
		*procOstream<<"XprocConduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
						<<"] constructed..."<<std::endl;
#endif
		}
		else
		{
			m_realConduitPtr = nullptr;
#ifdef USE_DEBUG_LOG
		PRINT_TIME_NOW(*procOstream)
		*procOstream<<"NULL Conduit: ["<<m_srcTask->getName()<<"<=>"<<m_dstTask->getName()
						<<"] constructed..."<<std::endl;
#endif
		}
	}
	else{
		std::cerr<<"ERROR, simple conduit only allow task rnning on one process"<<std::endl;
		return 1;
	}

	return 0;	
}

int Conduit::getCapacity()
{
	return m_capacity;
}

std::string Conduit::getName()
{
	return m_Name;
}

TaskBase* Conduit::getSrcTask()
{
    return m_srcTask;
}

TaskBase* Conduit::getDstTask()
{
    return m_dstTask;
}

TaskBase* Conduit::getAnotherTask()
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
		std::cerr<<"Error, conduit doesn't associated to calling task!"<<std::endl;
		exit(1);
	}

}

ConduitId_t Conduit::getConduitId()
{
    return m_conduitId;
}

void Conduit::Connect(TaskBase* srctask, TaskBase* dsttask)
{
    if(m_srcTask || m_dstTask)
    {
        std::cerr<<"Error, already connected to some Task"<<std::endl;
        exit(1);
    }
    
    initConduit(srctask, dsttask, INPROC_CONDUIT_CAPACITY_DEFAULT);
    
    return;
}


Conduit::~Conduit()
{
	// TODO: if conduit destruct too early so that task-threads still using cdt, this
	// may cause problem. Should handle this properly.
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
	}*/

	// delete this conduit item from conduit registry
	m_cdtMgr->unregisterConduit(this, m_conduitId);

	// after make sure that all task threads have finished,
	// destroy the real conduit object
	if(m_realConduitPtr)
	{
		delete m_realConduitPtr;
	}


#ifdef USE_DEBUG_LOG
    std::ofstream* procOstream = getProcOstream();
    PRINT_TIME_NOW(*procOstream)
    if(m_srcTask && m_dstTask)
    {
        *procOstream<<"Conduit: ["<<m_Name<<"] destroyed on proc "<<TaskManager::getRootTask()->getCurrentProcRank()<<"!!!"<<std::endl;
    }
    else
    {
        *procOstream<<"Conduit: [dummy conduit] destroyed on proc "<<TaskManager::getRootTask()->getCurrentProcRank()<<"!!!"<<std::endl;
    }
#endif

}


int Conduit::Write(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->Write(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

int Conduit::WriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->WriteBy(thread, DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

#ifdef ENABLE_OPBY_FINISH
void Conduit::WriteBy_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->WriteBy_Finish(tag);
	}

	return;
}
#endif

int Conduit::WriteByFirst(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr){
		m_realConduitPtr->WriteByFirst(DataPtr, DataSize, tag);
	}
	else{
		return 1;
	}
	return 0;
}


int Conduit::BWrite(void *DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->BWrite(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

int Conduit::BWriteBy(ThreadRank_t thread, void *DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->BWriteBy(thread, DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

#ifdef ENABLE_OPBY_FINISH
void Conduit::BWriteBy_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->BWriteBy_Finish(tag);
	}

	return;
}
#endif


int Conduit::PWrite(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->PWrite(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}


int Conduit::PWriteBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->PWriteBy(thread, DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

#ifdef ENABLE_OPBY_FINISH
void Conduit::PWriteBy_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->PWriteBy_Finish(tag);
	}

	return;
}
#endif


int Conduit::Read(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->Read(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

int Conduit::ReadBy(ThreadRank_t thread, void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->ReadBy(thread, DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

#ifdef ENABLE_OPBY_FINISH
void Conduit::ReadBy_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->ReadBy_Finish(tag);
	}

	return;
}
#endif

int Conduit::ReadByFirst(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr){
		m_realConduitPtr->Read(DataPtr, DataSize, tag);
	}
	else{
		return 1;
	}
	return 0;
}



int Conduit::AsyncRead(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->AsyncRead(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

void Conduit::AsyncRead_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->AsyncRead_Finish(tag);
	}

	return;
}


int Conduit::AsyncWrite(void* DataPtr, DataSize_t DataSize, int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->AsyncWrite(DataPtr, DataSize, tag);
	}
	else
	{
		return 1;
	}

	return 0;
}

void Conduit::AsyncWrite_Finish(int tag)
{
	if(m_realConduitPtr)
	{
		m_realConduitPtr->AsyncWrite_Finish(tag);
	}

	return;
}

}// end namespace iUtc
