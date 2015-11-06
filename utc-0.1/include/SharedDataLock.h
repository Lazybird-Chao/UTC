#ifndef UTC_SHARED_DATA_LOCK_H_
#define UTC_SHARED_DATA_LOCK_H_

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

namespace iUtc{

/*
 * supply a lock object that can be used in user task code.
 * To control access data that is shared by multiple task threads of one process.
 */

class SharedDataLock
{
public:
	SharedDataLock();

	~SharedDataLock();

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
	boost::unique_lock<boost::shared_mutex> m_Wlock{m_SharedMutex, boost::defer_lock};
	boost::shared_lock<boost::shared_mutex> m_Rlock{m_SharedMutex, boost::defer_lock};
};




/*
 *  A pair of macros, used in user task codes to define a piece of statements that will
 *  only be executed by one thread. Other threads would wait the completion of
 *  the executed thread.
 */

#define UNIQUE_EXE_START(onec_flag_idx)

#define UNIQUE_EXE_END


}



#endif
