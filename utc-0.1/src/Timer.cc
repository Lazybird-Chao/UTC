#include "Timer.h"
#include "Timer_Utilities.h"

#include <iostream>
#include <thread>

namespace iUtc{

Timer::Timer(Timer_unit tu)
{
	m_retTimeperiod =0.0;
	//m_started = false;
	switch(tu){
	case DAYS:
		m_ratio2sec = 1/86400;
		break;
	case HOURS:
		m_ratio2sec = 1/3600;
		break;
	case MINUTES:
		m_ratio2sec = 1/60;
		break;
	case SECONDS:
		m_ratio2sec = 1;
		break;
	case MILLISEC:
		m_ratio2sec = 1000;
		break;
	case MICROSEC:
		m_ratio2sec = 1000000;
		break;
	case NANOSEC:
		m_ratio2sec = 1000000000;
		break;
	default:
		m_ratio2sec = 1;
		break;
	}
	m_ratio2sec= m_ratio2sec*TIME_CLOCK::period::num / TIME_CLOCK::period::den;

}

void Timer::start()
{
	m_beginTimePoint = TIME_CLOCK::now();
	//m_started = true;
}
void Timer::start(TimerValue& tv_out)
{
	tv_out = m_beginTimePoint = TIME_CLOCK::now();
}

double Timer::stop()
{
	m_endTimePoint = TIME_CLOCK::now();
	/*if(m_started)
	{
		m_retTimevalue = (std::chrono::duration_cast<std::chrono::duration<double>>\
				(m_endTimePoint-m_beginTimePoint).count())*m_ratio2sec;

	}*/
	m_retTimeperiod=m_ratio2sec * (m_endTimePoint-m_beginTimePoint).count();
	return m_retTimeperiod;
}
double Timer::stop(TimerValue &tv_out)
{
	tv_out = m_endTimePoint = TIME_CLOCK::now();
	m_retTimeperiod=m_ratio2sec * (m_endTimePoint-m_beginTimePoint).count();
	return m_retTimeperiod;
}
double Timer::stop(TimerValue& tv_in, TimerValue& tv_out)
{
	tv_out = m_endTimePoint = TIME_CLOCK::now();
	m_retTimeperiod=m_ratio2sec * (m_endTimePoint-tv_in).count();
	return m_retTimeperiod;
}




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
