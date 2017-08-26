/*
 * helper_timer.h
 *
 *  Created on: Jan 13, 2017
 *      Author: chao
 */

#ifndef HELPER_TIMER_H_
#define HELPER_TIMER_H_

#include <time.h>

#define CLOCKTYPE CLOCK_REALTIME
//#define CLOCKTYPE CLOCK_MONOTONIC


double getTime(void){
	struct timespec curtime;
	clock_gettime(CLOCKTYPE, &curtime);
	return (double)curtime.tv_sec + (double)curtime.tv_nsec/1e9;
}



#endif /* HELPER_TIMER_H_ */
