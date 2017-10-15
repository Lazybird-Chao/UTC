/*
 * nbody_task.h
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_UTC_CPU_NBODY_TASK_H_
#define BENCHAPPS_NBODY_UTC_CPU_NBODY_TASK_H_

#include "Utc.h"
#include "UtcGpu.h"
#define MAX_TIMER 9

template<typename T>
class BodySystem: public UserTaskBase{
private:
	unsigned int m_numBodies;
	T *m_pos;
	T *m_vel;

	T m_softeningSquared;
	T m_damping;
	T *newPosBuffer;

	int process_numBodies;
	int process_startBodyIndex;

public:
	void initImpl(unsigned int numBodies,
			T softeningSquared,
			T damping,
			T *pos,
			T *vel);

	void runImpl(double runtime[][MAX_TIMER],
			int loops,
			int outInterval,
			int blocksize,
			T deltaTime,
			T *outbuffer);

};



#endif /* BENCHAPPS_NBODY_UTC_CPU_NBODY_TASK_H_ */
