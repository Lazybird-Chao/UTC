/*
 * AffinityConfig.h
 *
 *  Created on: Oct 21, 2017
 *      Author: Chao
 *
 */

#ifndef INCLUDE_AFFINITYCONFIG_H_
#define INCLUDE_AFFINITYCONFIG_H_

/*
 *  This file specify the current system's CPU info.
 *  For different platform, we may need modify this file manually currently.
 *
 *  TODO: For different platform, dynamically check system infomation and set the
 *  necessary data structure here.
 *
 */
#include "FastMutex.h"
#include <cstring>
#include <vector>

class Machine_CPU_info_t{
private:
	int m_total_cpus;
	int m_cores_per_cpu;
	int m_ht_per_core; //hyperthread

	int m_total_ht;

	int m_bind_mode;
	int m_dis_mode;	//
	int m_ht_mode;

	/*
	 * name cpu_id my be confusing, but linux system treat each ht as a cpu
	 * the size here is only a estimate max value, should change if necessary
	 */
	int m_cpu_ids[4][16][2];

	FastMutex m_next_cpu_mutex;
	int m_next_cpu;
	int m_next_cpu_active_count;
	int m_next_cpu_htid;

	FastMutex m_next_core_mutex;
	int m_next_core;
	int m_next_core_cpuid;
	int m_next_core_htid;

	FastMutex m_next_ht_mutex;
	int m_next_ht;
	int m_next_ht_cpuid;
	int m_next_ht_coreid;

public:
	/*
	 * currently manually change and set info here
	 */
	Machine_CPU_info_t(){
		/*
		 *
		 */
		m_total_cpus = 2;
		m_cores_per_cpu = 12;
		m_ht_per_core = 2;
		m_total_ht = m_total_cpus*m_cores_per_cpu*m_ht_per_core;

		/*
		 *  0: no bind, bind to all available cpus,cores,hardwarethreads
		 *
		 *  1: bind to single cpu
		 *
		 *  2: bind to sigle core
		 *
		 *  3: bind to single hyperthread
		 */
		m_bind_mode = 1;

		/*
		 *  0: choose core from next cpu alternatively
		 *
		 *  1: choose core from same cpu, until reach cores-per-cpu, then choose core from
		 *  	next cpu
		 */
		m_dis_mode = 0;

		/*
		 * 0: do not consider hyperthread, the basic resource unit is core
		 *
		 * 1: treat hyperthread as a basic resource unit
		 */
		m_ht_mode = 0;

		/*
		 *
		 */
		memset(m_cpu_ids, -1, sizeof(int)*m_total_ht);
		int cpu0[] = {0,1,2,3,4,5,6,7,8,9,10,11,
				24,25,26,27,28,29,30,31,32,33,34,35};
		int cpu1[] = {12,13,14,15,16,17,18,19,20,21,22,23,
				36,37,38,39,40,41,42,43,44,45,46,47};
		for(int i = 0; i< m_ht_per_core; i++){
			for(int j = 0; j< m_cores_per_cpu; j++)
				m_cpu_ids[0][j][i] = cpu0[i*m_cores_per_cpu+j];
		}
		for(int i = 0; i< m_ht_per_core; i++){
			for(int j = 0; j< m_cores_per_cpu; j++)
				m_cpu_ids[1][j][i] = cpu1[i*m_cores_per_cpu+j];
		}

		m_next_cpu = 0;
		m_next_cpu_active_count = 0;
		m_next_cpu_htid = 0;

		m_next_core = 0;
		m_next_core_cpuid = 0;
		m_next_core_htid = 0;

		m_next_ht = 0;
		m_next_ht_cpuid = 0;
		m_next_ht_coreid = 0;

	}

	std::vector<int> getCPUSET(){
		return getCPUSET(m_bind_mode);
	}

	std::vector<int> getCPUSET(int bind_mode){
		return getCPUSET(bind_mode, m_dis_mode);
	}

private:
	std::vector<int> getCPUSET(int bind_mode, int dis_mode){
		std::vector<int> cpuset;
		switch (bind_mode){
		case 0:
			break;
		case 1:
			bindToCPU(cpuset, dis_mode);
			break;
		case 2:
			bindToCore(cpuset, dis_mode);
			break;
		case 3:
			bindToHt(cpuset, dis_mode);
			break;
		default:
			break;
		}
		return cpuset;
	}

	void bindToCPU(std::vector<int> &cpuset, int dis_mode){
		m_next_cpu_mutex.lock();
		if(m_ht_mode == 0){
			if(dis_mode == 0){
				// always chose next cpu
				int cpuid = m_next_cpu;
				for(int j = 0; j<m_ht_per_core; j++)
					for(int i = 0; i<m_cores_per_cpu; i++)
						cpuset.push_back(m_cpu_ids[cpuid][i][j]);
				m_next_cpu = (m_next_cpu + 1) % m_total_cpus;
			}else if(dis_mode == 1){
				if(m_next_cpu_active_count == m_cores_per_cpu){
					m_next_cpu_active_count = 0;
					m_next_cpu = (m_next_cpu + 1) % m_total_cpus;
				}
				m_next_cpu_active_count++;
				int cpuid = m_next_cpu;
				for(int j = 0; j<m_ht_per_core; j++)
					for(int i = 0; i<m_cores_per_cpu; i++)
						cpuset.push_back(m_cpu_ids[cpuid][i][j]);
			}
		}else if(m_ht_mode == 1){
			if(dis_mode == 0){
				int cpuid = m_next_cpu;
				int htid = m_next_cpu_htid;
				for(int i = 0; i<m_cores_per_cpu; i++)
					cpuset.push_back(m_cpu_ids[cpuid][i][htid]);
				if(cpuid == m_total_cpus-1)
					m_next_cpu_htid = (m_next_cpu_htid + 1) % m_ht_per_core;
				m_next_cpu = (m_next_cpu + 1) % m_total_cpus;
			}else if(dis_mode == 1){
				if(m_next_cpu_active_count == m_cores_per_cpu*m_ht_per_core){
					m_next_cpu_active_count = 0;
					m_next_cpu = (m_next_cpu + 1) % m_total_cpus;
					m_next_cpu_htid = 0;
				}else if(m_next_cpu_active_count == m_cores_per_cpu)
					m_next_cpu_htid = (m_next_cpu_htid + 1) % m_ht_per_core;
				m_next_cpu_active_count++;
				int cpuid = m_next_cpu;
				int htid = m_next_cpu_htid;
				for(int i = 0; i<m_cores_per_cpu; i++)
					cpuset.push_back(m_cpu_ids[cpuid][i][htid]);
			}
		}
		m_next_cpu_mutex.unlock();
		return;
	}

	void bindToCore(std::vector<int> &cpuset, int dis_mode){
		m_next_core_mutex.lock();
		if(m_ht_mode == 0){
			if(dis_mode == 0){
				int cpuid = m_next_core_cpuid;
				int coreid = m_next_core;
				for(int i= 0; i<m_ht_per_core; i++)
					cpuset.push_back(m_cpu_ids[cpuid][coreid][i]);
				if(cpuid == m_total_cpus-1)
					m_next_core = (m_next_core + 1) % m_cores_per_cpu;
				m_next_core_cpuid = (m_next_core_cpuid + 1) % m_total_cpus;
			} else if(dis_mode == 1){
				int cpuid = m_next_core_cpuid;
				int coreid = m_next_core;
				for(int i= 0; i<m_ht_per_core; i++)
					cpuset.push_back(m_cpu_ids[cpuid][coreid][i]);
				if(m_next_core == m_cores_per_cpu-1)
					m_next_core_cpuid = (m_next_core_cpuid + 1) % m_total_cpus;
				m_next_core = (m_next_core + 1) % m_cores_per_cpu;
			}
		}else if(m_ht_mode == 1){
			if(dis_mode == 0){
				int cpuid = m_next_core_cpuid;
				int coreid = m_next_core;
				int htid = m_next_core_htid;
				cpuset.push_back(m_cpu_ids[cpuid][coreid][htid]);
				if(cpuid == m_total_cpus-1 && coreid == m_cores_per_cpu -1)
					m_next_core_htid = (m_next_core_htid +1) % m_ht_per_core;
				if(cpuid == m_total_cpus-1)
					m_next_core = (m_next_core + 1) % m_cores_per_cpu;
				m_next_core_cpuid = (m_next_core_cpuid + 1) % m_total_cpus;
			}else if(dis_mode == 1){
				int cpuid = m_next_core_cpuid;
				int coreid = m_next_core;
				int htid = m_next_core_htid;
				cpuset.push_back(m_cpu_ids[cpuid][coreid][htid]);
				if(coreid == m_cores_per_cpu-1 && htid == m_ht_per_core -1)
					m_next_core_cpuid = (m_next_core_cpuid + 1) % m_total_cpus;
				if(coreid == m_cores_per_cpu -1 )
					m_next_core_htid = (m_next_core_htid + 1) % m_ht_per_core;
				m_next_core = (m_next_core + 1) % m_cores_per_cpu;
			}
		}
		m_next_core_mutex.unlock();
		return;
	}

	void bindToHt(std::vector<int> &cpuset, int dis_mode){
		/*
		 * only with ht_mode == 1, otherwise should be same as bind to core
		 */
		m_next_ht_mutex.lock();
		if(m_ht_mode == 1){
			if(dis_mode == 0){
				int htid = m_next_ht;
				int coreid = m_next_ht_coreid;
				int cpuid = m_next_ht_cpuid;
				cpuset.push_back(m_cpu_ids[cpuid][coreid][htid]);
				if(cpuid == m_total_cpus -1 && htid == m_ht_per_core-1)
					m_next_ht_coreid = (m_next_ht_coreid + 1) % m_cores_per_cpu;
				if(cpuid == m_total_cpus-1)
					m_next_ht = (m_next_ht + 1) % m_ht_per_core;
				m_next_ht_cpuid = (m_next_ht_cpuid +1) % m_total_cpus;
			}else if(dis_mode == 1){
				int htid = m_next_ht;
				int coreid = m_next_ht_coreid;
				int cpuid = m_next_ht_cpuid;
				cpuset.push_back(m_cpu_ids[cpuid][coreid][htid]);
				if(htid == m_ht_per_core-1 && coreid == m_cores_per_cpu -1)
					m_next_ht_cpuid = (m_next_ht_cpuid + 1) % m_total_cpus;
				if(htid == m_ht_per_core -1)
					m_next_ht_coreid = (m_next_ht_coreid + 1) % m_cores_per_cpu;
				m_next_ht = (m_next_ht + 1) % m_ht_per_core;
			}
		}else if(m_ht_mode == 0){
			bindToCore(cpuset, dis_mode);
		}
		m_next_ht_mutex.unlock();
		return;
	}

};



















#endif /* INCLUDE_AFFINITYCONFIG_H_ */
