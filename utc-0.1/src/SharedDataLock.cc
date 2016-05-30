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
	long _counter=0;
	while(m_state.exchange(Locked, std::memory_order_acquire) == Locked){
		if(_counter<USE_PAUSE)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP){
			__asm__ __volatile__ ("pause" ::: "memory");
			std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);
	}
}

void SpinLock::unlock(){
	m_state.store(Unlocked, std::memory_order_release);
}

}// end namespace iUtc


