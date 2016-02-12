#ifndef UTC_TIMER_H_
#define UTC_TIMER_H_

#include <time.h>
#include <chrono>

#define TIME_CLOCK  std::chrono::steady_clock
//#define TIME_CLOCK  std::chrono::high_resolution_clock
//#define TIME_CLOCK  std::chrono::system_clock

namespace iUtc{

enum Timer_unit{
	DAYS =0,
	HOURS,
	MINUTES,
	SECONDS,
	MILLISEC,
	MICROSEC,
	NANOSEC
};
class Timer
{
public:

	typedef TIME_CLOCK::time_point TimerValue;

	Timer(Timer_unit tu=SECONDS);

	// start a timer obj, do a timer record
	void start();
	// tv_out store the current time point value
	void start(TimerValue& tv_out);

	// do a record on the timer obj, return the past time  from last start()
	// to current stop() in Timer_unit
	double stop();
	// return passed time from last start(), also tv_out store the current time point value
	double stop(TimerValue& tv_stop);
	// return the time period from tv_in to current. also tv_out store the current time point
	double stop(TimerValue& tv_start, TimerValue& tv_stop);

	double getRealTime();
#ifdef _LINUX_
	double getThreadCpuTime();
#endif

private:
	TIME_CLOCK::time_point m_beginTimePoint;
	TIME_CLOCK::time_point m_endTimePoint;

	double m_realTimeperiod;
	double m_ratio2sec;
#ifdef _LINUX_
	double m_cpuTimeperiod;

	/*
	 * some other linux timing struct
	 */
	struct timespec m_clk_start;
	struct timespec m_clk_stop;
#endif
	//bool m_started;

	Timer& operator=(const Timer &other)=delete;
	Timer(const Timer &other)= delete;

};


}




#endif

