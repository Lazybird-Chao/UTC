#include "SharedDataLock.h"
#include "UtcBasics.h"

namespace iUtc{

/*SharedDataLock::SharedDataLock()
:m_Wlock(m_SharedMutex, boost::defer_lock),
 m_Rlock(m_SharedMutex, boost::defer_lock)
{

}


SharedDataLock::~SharedDataLock()
{

}
*/

void SharedDataLock::lock()
{
	//m_Wlock.lock();
	m_SharedMutex.lock();
}

void SharedDataLock::unlock()
{
	//m_Wlock.unlock();
	m_SharedMutex.unlock();
}

void SharedDataLock::write_lock()
{
	//m_Wlock.lock();
	m_SharedMutex.lock();
}

void SharedDataLock::write_unlock()
{
	//m_Wlock.unlock();
	m_SharedMutex.unlock();
}

void SharedDataLock::read_lock()
{
	//m_Rlock.lock();
	m_SharedMutex.lock_shared();
}

void SharedDataLock::read_unlock()
{
	//m_Rlock.unlock();
	m_SharedMutex.unlock_shared();
}


void SharedDataLock::fastlock(){
	m_FastMutex.lock();
}

void SharedDataLock::fastunlock(){
	m_FastMutex.unlock();
}

}// end namespace iUtc


