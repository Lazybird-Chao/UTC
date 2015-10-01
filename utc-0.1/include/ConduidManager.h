#ifndef UTC_CONDUIT_MANAGER_H_
#define UTC_CONDUIT_MANAGER_H_

#include "UtcBasics.h"
#include "Conduit.h"

#include <map>
#include <mutex>
#include <thread>


namespace iUtc{

class ConduitManager{
public:
	static ConduitManager* getInstance();

	static ConduitId registerConduit(Conduit* cdt);

	static void unregisterConduit(Conduit* cdt);

	static ConduitId getNewConduitId();

	static int getNumConduits();

	~ConduitManager();

private:
	static ConduitManager* m_InstancePtr;

	static ConduitId m_ConduitIdDealer;

	static std::mutex m_mutexConduitRegistry;
	static std::mutex m_mutexConduitIdDealer;

	static std::map<ConduitId, Conduit*> m_ConduitRegistry;

	ConduitManager();

};


}// namespace iUtc





#endif


