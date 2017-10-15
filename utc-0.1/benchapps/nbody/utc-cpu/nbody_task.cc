/*
 * nbody_task.cc
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */

#include "nbody_task.h"
#include <iostream>
#define MAX_TIMER 9

template<typename T>
thread_local int BodySystem<T>::thread_numBodies;
template<typename T>
thread_local int BodySystem<T>::thread_startBodyIndex;

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
		oldPosBuffer = new T[process_numBodies*4];
		newPosBuffer = new T[process_numBodies*4];
	}
	__fastIntraSync.wait();
	int bodiesPerThread = numBodies/__numGlobalThreads;
	if(__globalThreadId < numBodies % __numGlobalThreads){
		thread_numBodies = bodiesPerThread +1;
		thread_startBodyIndex = __globalThreadId *(bodiesPerThread +1);
	}
	else{
		thread_numBodies = bodiesPerThread;
		thread_startBodyIndex = __globalThreadId * bodiesPerThread + numBodies%__numGlobalThreads;
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
		T deltaTime,
		T *outbuffer
		){
	if(__globalThreadId == 0)
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	Timer timer, timer0;
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;

	T* m_force = new T[3*thread_numBodies];

	/*
	 * bcast init body pos and vel to all nodes
	 */
	timer0.start();
	timer.start();
	TaskBcastBy<T>(this, m_pos, 4*m_numBodies, 0);
	TaskBcastBy<T>(this, m_vel, 4*m_numBodies, 0);
	__fastIntraSync.wait();
	commtime += timer.stop();

	int i = 0;
	while(i<loops){
		timer.start();
		memcpy(oldPosBuffer+__localThreadId*thread_numBodies*4,
				m_pos+4*thread_startBodyIndex, thread_numBodies*4*sizeof(T));
		computeNBodyGravitation(thread_startBodyIndex,
								thread_numBodies,
								m_force);
		integreateNBodySystem(deltaTime, m_force);
		computetime += timer.stop();
		__fastIntraSync.wait();
		/*
		 * gather new bodypos from all
		 */
		timer.start();
		TaskGatherBy<T>(this, newPosBuffer, process_numBodies*4, m_pos, process_numBodies*4, 0);
		TaskBcastBy<T>(this, m_pos, 4*m_numBodies, 0);
		__fastIntraSync.wait();
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
	delete m_force;
	if(__localThreadId ==0){
		delete oldPosBuffer;
		delete newPosBuffer;
	}
	if(__processIdInGroup == 0){
		runtime[__localThreadId][0] = totaltime;
		runtime[__localThreadId][1] = computetime;
		runtime[__localThreadId][2] = commtime;
	}
	if(__globalThreadId == 0)
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";


}

template<typename T>
void BodySystem<T>::computeNBodyGravitation(int start_bodyIdex, int numBodies, T* m_force){
	T * pos = oldPosBuffer + __localThreadId*thread_numBodies*4;
	for(int i = 0; i < thread_numBodies; i++){
		int indexForce = 3*i;
		T acc[3] = {0, 0, 0};

		// We unroll this loop 4X for a small performance boost.
		int j = 0;

		while (j < m_numBodies)
		{
			bodyBodyInteraction<T>(acc, &pos[4*i], &m_pos[4*j], m_softeningSquared);
			j++;
			bodyBodyInteraction<T>(acc, &pos[4*i], &m_pos[4*j], m_softeningSquared);
			j++;
			bodyBodyInteraction<T>(acc, &pos[4*i], &m_pos[4*j], m_softeningSquared);
			j++;
			bodyBodyInteraction<T>(acc, &pos[4*i], &m_pos[4*j], m_softeningSquared);
			j++;
		}

		m_force[indexForce  ] = acc[0];
		m_force[indexForce+1] = acc[1];
		m_force[indexForce+2] = acc[2];
	}

}

template<typename T>
void BodySystem<T>::integreateNBodySystem(T deltaTime, T *m_force){
	T * pos_old = oldPosBuffer + __localThreadId*thread_numBodies*4;
	T * new_pos = newPosBuffer + __localThreadId*thread_numBodies*4;
	T * vel_old = m_vel + 4*thread_startBodyIndex;
	for (int i = 0; i < thread_numBodies; ++i)
	{
		int index = 4*i;
		int indexForce = 3*i;


		T pos[3], vel[3], force[3];
		pos[0] = pos_old[index+0];
		pos[1] = pos_old[index+1];
		pos[2] = pos_old[index+2];
		T invMass = pos_old[index+3];

		vel[0] = vel_old[index+0];
		vel[1] = vel_old[index+1];
		vel[2] = vel_old[index+2];

		force[0] = m_force[indexForce+0];
		force[1] = m_force[indexForce+1];
		force[2] = m_force[indexForce+2];

		// acceleration = force / mass;
		// new velocity = old velocity + acceleration * deltaTime
		vel[0] += (force[0] * invMass) * deltaTime;
		vel[1] += (force[1] * invMass) * deltaTime;
		vel[2] += (force[2] * invMass) * deltaTime;

		vel[0] *= m_damping;
		vel[1] *= m_damping;
		vel[2] *= m_damping;

		// new position = old position + velocity * deltaTime
		pos[0] += vel[0] * deltaTime;
		pos[1] += vel[1] * deltaTime;
		pos[2] += vel[2] * deltaTime;

		new_pos[index+0] = pos[0];
		new_pos[index+1] = pos[1];
		new_pos[index+2] = pos[2];
		new_pos[index+3] = pos_old[index+3];


		vel_old[index+0] = vel[0];
		vel_old[index+1] = vel[1];
		vel_old[index+2] = vel[2];
	}

}

template class BodySystem<double>;
template class BodySystem<float>;


















