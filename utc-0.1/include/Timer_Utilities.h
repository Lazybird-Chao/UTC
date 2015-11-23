#ifndef UTC_TIMER_UTILITIES_H_
#define UTC_TIMER_UTILITIES_H_

#include "UtcBasics.h"
#include <chrono>

namespace iUtc{

void sleep_for(long n_seconds);

void msleep_for(long n_mseconds);

void usleep_for(long n_useconds);

double time_from_start();


}



#endif

