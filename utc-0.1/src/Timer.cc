#include "Timer.h"
#include <iostream>

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
	m_ratio2sec= m_ratio2sec*std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

}

void Timer::start()
{
	m_beginTimePoint = std::chrono::steady_clock::now();
	//m_started = true;
}
void Timer::start(TimerValue& tv_out)
{
	tv_out = m_beginTimePoint = std::chrono::steady_clock::now();
}

double Timer::stop()
{
	m_endTimePoint = std::chrono::steady_clock::now();
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
	tv_out = m_endTimePoint = std::chrono::steady_clock::now();
	m_retTimeperiod=m_ratio2sec * (m_endTimePoint-m_beginTimePoint).count();
	return m_retTimeperiod;
}
double Timer::stop(TimerValue& tv_in, TimerValue& tv_out)
{
	tv_out = m_endTimePoint = std::chrono::steady_clock::now();
	m_retTimeperiod=m_ratio2sec * (m_endTimePoint-tv_in).count();
	return m_retTimeperiod;
}

}
