/*
 * SpinLock.cc
 *
 *  Created on: Nov 28, 2016
 *      Author: chao
 */

#include "SpinLock.h"
#include "UtcBasics.h"
#include "TimerUtilities.h"


namespace iUtc{

SpinLock::SpinLock(){
	m_state = Unlocked;
}

void SpinLock::lock(int id){
	long _counter=0;
	while(m_state.exchange(Locked, std::memory_order_acquire) == Locked){
		_counter++;
		/*if(_counter<USE_PAUSE + id*100)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP + id*200){
			__asm__ __volatile__ ("pause" ::: "memory");
			if(_counter % 1000 ==0)
				std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);
			*/
		spinWait(_counter, id);
	}
}

void SpinLock::unlock(){
	m_state.store(Unlocked, std::memory_order_release);
}

}


