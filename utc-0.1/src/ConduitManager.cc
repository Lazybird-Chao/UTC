#include "ConduitManager.h"
#include "UtcBasics.h"


namespace iUtc{

ConduitManager* ConduitManager::m_InstancePtr = nullptr;
ConduitId ConduitManager::m_ConduitIdDealer = 0;
std::mutex ConduitManager::m_mutexConduitRegistry;
std::mutex ConduitManager::m_mutexConduitIdDealer;
std::map<Conduit*, ConduitId> ConduitManager::m_ConduitRegistry;

ConduitManager::ConduitManger(){}

ConduitManager::~ConduitManager()
{
	if(m_InstancePtr)
	{
		m_ConduitRegistry.clear();
		m_InstancePtr = nullptr;
		m_ConduitIdDealer = 0;
	}
}

ConduitManager::getInstance()
{
	// Singleton instance
	if(!m_InstancePtr)
	{
		// as we only call this function at very beginning, at context startup
		// there should not be multi-thread issue
		// also, with c++11, the "static" local var would ensure only one thread define the obj
		static ConduitManager cdtMgr;
		m_InstancePtr = &cdtMgr;
		m_ConduitIdDealer = 1;   // conduit id starts from 1
	}
	return m_InstancePtr;
}

ConduitId ConduitManager::registerConduit(Conduit* cdt)
{
	std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
	TaskId  id = cdt->getConduitId();
	m_ConduitRegistry.insert(std::pair<ConduitId, Conduit*>(id, cdt));
	return id;
}

void ConduitManager::unregisterConduit(Conduit* cdt)
{
	std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
	ConduitId id = cdt->getConduitId();
	if(id)
	{
		m_TaskRegistry.erase(id);
	}
	return;

}

ConduitId ConduitManager::getNewConduitId()
{
	std::lock_guard<std::mutex> lock(m_mutexConduitIdDealer);
	ConduitId id = m_ConduitIdDealer++;
	return id;

}

int ConduitManager::getNumConduits()
{
	return m_TaskRegistry.size();
}



}// namespace iUtc
