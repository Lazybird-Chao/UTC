#ifndef UTC_TIMER_H_
#define UTC_TIMER_H_

#include <chrono>


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

	typedef std::chrono::steady_clock::time_point TimerValue;

	Timer(Timer_unit tu=SECONDS);

	// start a timer obj, do a timer record
	void start();
	// tv_out store the current time point value
	void start(TimerValue& tv_out);

	// do a record on the timer obj, return the past time  from last start()
	// to current stop() in Timer_unit
	double stop();
	// return passed time from last start(), also tv_out store the current time point value
	double stop(TimerValue& tv_out);
	// return the time period from tv_in to current. also tv_out store the current time point
	double stop(TimerValue& tv_in, TimerValue& tv_out);



private:
	std::chrono::steady_clock::time_point m_beginTimePoint;
	std::chrono::steady_clock::time_point m_endTimePoint;

	double m_retTimeperiod;
	double m_ratio2sec;

	//bool m_started;

	Timer& operator=(const Timer &other)=delete;
	Timer(const Timer &other)= delete;

};


}




#endif

