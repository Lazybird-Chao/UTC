/*
 * AffinityUtilities.h
 *
 *  Created on: Feb 6, 2016
 *      Author: chao
 */

#ifndef UTC_AFFINITYUTILITIES_H_
#define UTC_AFFINITYUTILITIES_H_

#ifdef _LINUX_
	#include <unistd.h>
	#include <sys/types.h>
	#include <sys/syscall.h>
#endif
#ifdef _MAC_

#endif
#include <vector>

namespace iUtc{

/*
 * get the cpu affinity configuration of calling thread
 */
inline std::vector<int> getAffinityLinux(){
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

inline std::vector<int> getAffinityMac(){
	return std::vector<int>(0);
}


inline std::vector<int> getAffinity()
{
#ifdef _LINUX_
	return getAffinityLinux();
#endif
#ifdef _MAC_
	return getAffinityMAc();
#endif

}



/*
 * modify calling thread's cpu affinity as desired
 */
inline void setAffinityLinux(std::vector<int> cpus){
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

inline void setAffinityMac(std::vector<int> cpus){
	return;
}


inline void setAffinity(std::vector<int> cpus)
{
#ifdef _LINUX_
	setAffinityLinux(cpus);
#endif
#ifdef _MAC_
	setAffinityMac(cpus);
#endif
}

/*
 * get calling thread's run-on cpu id at current execution point
 */
inline int getCurrentCPU()
{
#ifdef _LINUX_
	return sched_getcpu();
#endif

#ifdef _MAC_
	return 0;
#endif
}

}//end iUtc







#endif /* INCLUDE_AFFINITYUTILITIES_H_ */
