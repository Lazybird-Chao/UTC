#include "Timer.h"
#include <iostream>
#include <thread>
#include "../include/TimerUtilities.h"

namespace iUtc{

Timer::Timer(Timer_unit tu)
{
	m_realTimeperiod =0.0;
#ifdef _LINUX_
	m_cpuTimeperiod =0.0;
#endif
	//m_started = false;
	switch(tu){
	case DAYS:
		m_ratio2sec = 1.0/86400;
		break;
	case HOURS:
		m_ratio2sec = 1.0/3600;
		break;
	case MINUTES:
		m_ratio2sec = 1.0/60;
		break;
	case SECONDS:
		m_ratio2sec = 1.0;
		break;
	case MILLISEC:
		m_ratio2sec = 1000.0;
		break;
	case MICROSEC:
		m_ratio2sec = 1000000.0;
		break;
	case NANOSEC:
		m_ratio2sec = 1000000000.0;
		break;
	default:
		m_ratio2sec = 1.0;
		break;
	}
	m_ratio2sec= m_ratio2sec*TIME_CLOCK::period::num / TIME_CLOCK::period::den;
	m_beginTimePoint = TIME_CLOCK::time_point();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_start);
#endif

}

void Timer::start()
{
	m_beginTimePoint = TIME_CLOCK::now();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_start);
#endif
	//m_started = true;
}
void Timer::start(TimerValue& tv_out)
{
	tv_out = m_beginTimePoint = TIME_CLOCK::now();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_start);
#endif
}

double Timer::stop()
{
	m_endTimePoint = TIME_CLOCK::now();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_stop);
#endif
	/*if(m_started)
	{
		m_retTimevalue = (std::chrono::duration_cast<std::chrono::duration<double>>\
				(m_endTimePoint-m_beginTimePoint).count())*m_ratio2sec;

	}*/
	m_realTimeperiod=m_ratio2sec * (m_endTimePoint-m_beginTimePoint).count();

	return m_realTimeperiod;
}
double Timer::stop(TimerValue &tv_out)
{
	tv_out = m_endTimePoint = TIME_CLOCK::now();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_stop);
#endif
	m_realTimeperiod=m_ratio2sec * (m_endTimePoint-m_beginTimePoint).count();
	return m_realTimeperiod;
}
double Timer::stop(TimerValue& tv_in, TimerValue& tv_out)
{
	tv_out = m_endTimePoint = TIME_CLOCK::now();
#ifdef _LINUX_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_clk_stop);
#endif
	m_realTimeperiod=m_ratio2sec * (m_endTimePoint-tv_in).count();
	return m_realTimeperiod;
}

/*
 *
 */
double Timer::getRealTime()
{
	return m_realTimeperiod;
}

/*
 * get calling thread's own cpu time, excluding waiting for other thread's cpu holding time.
 * this is not the real feeling or wall clock time.
 */
#ifdef _LINUX_
double Timer::getThreadCpuTime()
{
	m_cpuTimeperiod = (m_ratio2sec*TIME_CLOCK::period::den/TIME_CLOCK::period::num)*
			((m_clk_stop.tv_sec - m_clk_start.tv_sec) + (m_clk_stop.tv_nsec - m_clk_start.tv_nsec)/1e9);
	return m_cpuTimeperiod;
}
#endif


/*
 *  Some time utility functions
 */

void sleep_for(long n_seconds)
{
	std::this_thread::sleep_for(std::chrono::seconds(n_seconds));
}

void msleep_for(long n_mseconds)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(n_mseconds));

}

void usleep_for(long n_useconds)
{
	std::this_thread::sleep_for(std::chrono::microseconds(n_useconds));
}

// return the time period in milliseconds from system start to current
double time_from_start()
{
	 std::chrono::system_clock::time_point t_now=std::chrono::system_clock::now();
	 //std::chrono::system_clock::duration dtn =t_now-SYSTEM_START_TIME;
	 //return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1000>>>(dtn).count();
	 std::chrono::duration<double, std::ratio<1,1000>> dtn = t_now-SYSTEM_START_TIME;
	 return dtn.count();
}


}// end namespace iUtc
