/*
 * SpinLock.h
 *
 *  Created on: Nov 28, 2016
 *      Author: chao
 */

#ifndef UTC_SPINLOCK_H_
#define UTC_SPINLOCK_H_

#include <atomic>

namespace iUtc{

class SpinLock{
public:
	SpinLock();

	void lock(int id=0);
	void unlock();
private:
	typedef enum{Locked, Unlocked} LockState;
	std::atomic<LockState> m_state;
};

}





#endif /* SPINLOCK_H_ */
