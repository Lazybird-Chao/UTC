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



SpinLock::SpinLock(){
	m_state = Unlocked;
}

void SpinLock::lock(){
	while(m_state.exchange(Locked, std::memory_order_acquire) == Locked){
		_mm_pause();
	}
}

void SpinLock::unlock(){
	m_state.store(Unlocked, std::memory_order_release);
}

}// end namespace iUtc


