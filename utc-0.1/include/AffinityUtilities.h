/*
 * AffinityUtilities.h
 *
 *  Created on: Feb 6, 2016
 *      Author: chao
 */

#ifndef UTC_AFFINITYUTILITIES_H_
#define UTC_AFFINITYUTILITIES_H_

#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <vector>

namespace iUtc{

/*
 * get the cpu affinity configuration of calling thread
 */
inline std::vector<int> getAffinity()
{
	cpu_set_t cpuset;
	pthread_t thread;
	std::vector<int> ret;
	CPU_ZERO(&cpuset);
	thread = pthread_self();
	int s;
	s= pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	if(s)
	{
		std::cerr<<"ERROR, Affinity get error!"<<std::endl;
		return ret;
	}
	for(int i=0; i<CPU_SETSIZE; i++)
		if(CPU_ISSET(i, &cpuset))
			ret.push_back(i);
	return ret;
}


/*
 * modify calling thread's cpu affinity as desired
 */
inline void setAffinity(std::vector<int> cpus)
{
	cpu_set_t cpuset;
	pthread_t thread;
	std::vector<int> ret;
	CPU_ZERO(&cpuset);
	thread = pthread_self();
	for(int i=0; i<cpus.size(); i++)
		CPU_SET(cpus[i], &cpuset);
	int s;
	s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	if(s)
	{
		std::cerr<<"ERROR, Affinity set error!"<<std::endl;
		return;
	}

	return;
}

/*
 * get calling thread's run-on cpu id at current execution point
 */
inline int getCurrentCPU()
{
	return sched_getcpu();
}

}//end iUtc







#endif /* INCLUDE_AFFINITYUTILITIES_H_ */
