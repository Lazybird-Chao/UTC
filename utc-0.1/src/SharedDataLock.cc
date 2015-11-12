#include "SharedDataLock.h"

namespace iUtc{

SharedDataLock::SharedDataLock()
:m_Wlock(m_SharedMutex, boost::defer_lock),
 m_Rlock(m_SharedMutex, boost::defer_lock)
{

}


SharedDataLock::~SharedDataLock()
{

}

void SharedDataLock::lock()
{
	m_Wlock.lock();
}

void SharedDataLock::unlock()
{
	m_Wlock.unlock();
}

void SharedDataLock::write_lock()
{
	m_Wlock.lock();
}

void SharedDataLock::write_unlock()
{
	m_Wlock.unlock();
}

void SharedDataLock::read_lock()
{
	m_Rlock.lock();
}

void SharedDataLock::read_unlock()
{
	m_Rlock.unlock();
}


}// end namespace iUtc


