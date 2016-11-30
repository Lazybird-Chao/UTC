#ifndef UTC_TIMER_UTILITIES_H_
#define UTC_TIMER_UTILITIES_H_

#include "UtcBasics.h"
#include <emmintrin.h>
#include <chrono>

namespace iUtc{

void sleep_for(long n_seconds);

void msleep_for(long n_mseconds);

void usleep_for(long n_useconds);

double time_from_start();

inline void spinWait(long _counter, int id=0){
	if(_counter<USE_PAUSE + id*100)
		_mm_pause();
	else if(_counter<USE_SHORT_SLEEP + id*200){
		__asm__ __volatile__ ("pause" ::: "memory");
		if(_counter %1000 ==0)
			std::this_thread::yield();
	}
	else if(_counter<USE_LONG_SLEEP)
		nanosleep(&SHORT_PERIOD, nullptr);
	else
		nanosleep(&LONG_PERIOD, nullptr);
}

}



#endif

