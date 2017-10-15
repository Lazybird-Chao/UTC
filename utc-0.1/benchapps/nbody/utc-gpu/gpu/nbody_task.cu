/*
 * nbody_task.cc
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */

#include "nbody_task.h"
#include "nbody_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>
#define MAX_TIMER 9

template<typename T>
void BodySystem<T>::initImpl(
				unsigned int numBodies,
				T softeningSquared,
				T damping,
				T *pos,
				T *vel){
	if(__localThreadId == 0){
		m_numBodies = numBodies;
		m_softeningSquared = softeningSquared;
		m_damping = damping;
		if(__processIdInGroup == 0){
			m_pos = pos;
			m_vel = vel;
		}else{
			m_pos = new T[4*numBodies];
			m_vel = new T[4*numBodies];
		}
		process_numBodies = m_numBodies/__numGroupProcesses;
		process_startBodyIndex = __processIdInGroup * process_numBodies;
		newPosBuffer = new T[process_numBodies*4];
	}
	inter_Barrier();
	if(__globalThreadId == 0)
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
}

template<typename T>
void BodySystem<T>::runImpl(
		double runtime[][MAX_TIMER],
		int loops,
		int outInterval,
		int blocksize,
		T deltaTime,
		T *outbuffer
		){
	if(__globalThreadId == 0)
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	Timer timer, timer0;
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;
	double copytime = 0;

	GpuData<T> m_pos_d(m_numBodies*4);
	GpuData<T> new_pos_d(process_numBodies*4);
	GpuData<T> m_vel_d(process_numBodies*4);

	int mingridsize = 1;
	double kernelTime =0;
	double copyoutTime = 0;
	dim3 block(blocksize, 1,1);
	int threadsperBody = 1;
	if(process_numBodies/blocksize >= mingridsize)
		threadsperBody = 1;
	else
		threadsperBody = blocksize*mingridsize/process_numBodies;  //should keep this dividable
	dim3 grid((process_numBodies*threadsperBody+blocksize-1)/blocksize, 1,1);
	int ntiles = m_numBodies/block.x;

	/*
	 * bcast init body pos and vel to all nodes
	 */
	timer0.start();
	timer.start();
	TaskBcastBy<T>(this, m_pos, 4*m_numBodies, 0);
	TaskBcastBy<T>(this, m_vel, 4*m_numBodies, 0);
	//__fastIntraSync.wait();
	commtime += timer.stop();
	timer.start();
	new_pos_d.putD(m_pos + process_startBodyIndex*4);
	m_vel_d.putD(m_vel + process_startBodyIndex*4);
	copytime += timer.stop();

	int i = 0;
	while(i<loops){
		timer.start();
		m_pos_d.putD(m_pos);
		copytime += timer.stop();
		timer.start();
		if(threadsperBody>1){
			_integrateNBodySystemSmall_kernel<T><<<grid, block, 0, __streamId>>>(
							m_pos_d.getD(),
							new_pos_d.getD(true),
							m_vel_d.getD(true),
							m_numBodies,
							deltaTime,
							m_softeningSquared,
							m_damping,
							ntiles,
							threadsperBody,
							process_numBodies,
							process_startBodyIndex);
		}
		else{
			_integrateNBodySystem_kernel<T><<<grid, block, 0, __streamId>>>(
						m_pos_d.getD(),
						new_pos_d.getD(true),
						m_vel_d.getD(true),
						m_numBodies,
						deltaTime,
						m_softeningSquared,
						m_damping,
						ntiles,
						process_numBodies,
						process_startBodyIndex);
		}
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaStreamSynchronize(__streamId));
		computetime += timer.stop();

		timer.start();
		new_pos_d.fetchD(newPosBuffer);
		copytime += timer.stop();
		/*
		 * gather new bodypos from all
		 */
		timer.start();
		TaskGatherBy<T>(this, newPosBuffer, process_numBodies*4, m_pos, process_numBodies*4, 0);
		TaskBcastBy<T>(this, m_pos, 4*m_numBodies, 0);
		//__fastIntraSync.wait();
		commtime += timer.stop();

		i++;
		if(i%outInterval ==0 && __globalThreadId == 0){
			int offset = (i/outInterval -1)*m_numBodies*4;
			memcpy(outbuffer+offset, m_pos,
					m_numBodies*4*sizeof(T));
		}
	}
	inter_Barrier();
	totaltime = timer0.stop();
	if(__localThreadId ==0){
		delete newPosBuffer;
	}
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = commtime;
		runtime[__localThreadId][3] = copytime;
	}
	if(__globalThreadId == 0)
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";


}

template class BodySystem<double>;
template class BodySystem<float>;


















