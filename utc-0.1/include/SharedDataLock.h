#ifndef UTC_SHARED_DATA_LOCK_H_
#define UTC_SHARED_DATA_LOCK_H_

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <atomic>

namespace iUtc{

/*
 * supply a lock object that can be used in user task code.
 * To control access data that is shared by multiple task threads of one process.
 */

class SharedDataLock
{
public:
	//SharedDataLock();

	//~SharedDataLock();

	// standard unique lock, one thread get lock at one time
	void lock();
	void unlock();

	// multi threads could get lock at same time
	void read_lock();
	void read_unlock();

	// similar as lock()
	void write_lock();
	void write_unlock();


private:
	boost::shared_mutex m_SharedMutex;
	//boost::unique_lock<boost::shared_mutex> m_Wlock{m_SharedMutex, boost::defer_lock};
	//boost::shared_lock<boost::shared_mutex> m_Rlock{m_SharedMutex, boost::defer_lock};
};




class SpinLock{
public:
	SpinLock();

	void lock();
	void unlock();
private:
	typedef enum{Locked, Unlocked} LockState;
	std::atomic<LockState> m_state;
};

}// end namespace iUtc

#endif
