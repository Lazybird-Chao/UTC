#include "ConduitManager.h"
#include "Conduit.h"
#include "UtcBasics.h"
#include <map>
#include <cassert>



namespace iUtc{

ConduitManager* ConduitManager::m_InstancePtr = nullptr;
ConduitId_t ConduitManager::m_ConduitIdDealer = 0;
std::mutex ConduitManager::m_mutexConduitRegistry;
std::mutex ConduitManager::m_mutexConduitIdDealer;
std::map<ConduitId_t, Conduit*> ConduitManager::m_ConduitRegistry;

std::ofstream* getProcOstream();

ConduitManager::ConduitManager(){}

ConduitManager::~ConduitManager()
{
	if(m_InstancePtr)
	{
#ifdef USE_DEBUG_ASSERT
	    assert(m_ConduitRegistry.size() == 0);
#endif
		m_ConduitRegistry.clear();
		m_InstancePtr = nullptr;
		m_ConduitIdDealer = 0;
	}
#ifdef USE_DEBUG_LOG

#endif
}

ConduitManager* ConduitManager::getInstance()
{
	// Singleton instance
	if(!m_InstancePtr)
	{
		// as we only call this function at very beginning, at context startup
		// there should not be multi-thread issue
		// also, with c++11, the "static" local var would ensure only one thread define the obj
		// means there's only one var, all threads shared this,
	    static ConduitManager cdtMgr;
		m_InstancePtr = &cdtMgr;
		m_ConduitIdDealer = 1;   // conduit id starts from 1
	}
	return m_InstancePtr;
}

ConduitId_t ConduitManager::registerConduit(Conduit* cdt)
{
	std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
	TaskId_t  id = cdt->getConduitId();
	m_ConduitRegistry.insert(std::pair<ConduitId_t, Conduit*>(id, cdt));
	return id;
}
void ConduitManager::registerConduit(Conduit* cdt, int id)
{
    std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
    m_ConduitRegistry.insert(std::pair<ConduitId_t, Conduit*>(id, cdt));
    return;
}

void ConduitManager::unregisterConduit(Conduit* cdt)
{
	std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
	ConduitId_t id = cdt->getConduitId();
	if(id)
	{
		m_ConduitRegistry.erase(id);   // should check if in the map
	}
	return;

}
void ConduitManager::unregisterConduit(Conduit* cdt, int id)
{
    std::lock_guard<std::mutex> lock(m_mutexConduitRegistry);
    m_ConduitRegistry.erase(id);     // should check if in the map
    return;

}

ConduitId_t ConduitManager::getNewConduitId()
{
	std::lock_guard<std::mutex> lock(m_mutexConduitIdDealer);
	ConduitId_t id = m_ConduitIdDealer++;
	return id;

}

int ConduitManager::getNumConduits()
{
	return m_ConduitRegistry.size();
}



}// namespace iUtc
