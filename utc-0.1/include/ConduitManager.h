#ifndef UTC_CONDUIT_MANAGER_H_
#define UTC_CONDUIT_MANAGER_H_

#include "UtcBasics.h"
//#include "Conduit.h"

#include <map>
#include <mutex>
#include <thread>


namespace iUtc{

// pre declaration
class Conduit;

class ConduitManager{
public:
	static ConduitManager* getInstance();

	static ConduitId_t registerConduit(Conduit* cdt);
	static void registerConduit(Conduit* cdt, int id);

	static void unregisterConduit(Conduit* cdt);
	static void unregisterConduit(Conduit* cdt, int id);

	static ConduitId_t getNewConduitId();

	static int getNumConduits();

	~ConduitManager();

private:
	static ConduitManager* m_InstancePtr;

	static ConduitId_t m_ConduitIdDealer;

	static std::mutex m_mutexConduitRegistry;
	static std::mutex m_mutexConduitIdDealer;

	static std::map<ConduitId_t, Conduit*> m_ConduitRegistry;

	ConduitManager();

};


}// namespace iUtc





#endif


