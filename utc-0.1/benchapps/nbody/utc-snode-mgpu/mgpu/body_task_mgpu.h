/*
 * bodysystem.h
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_UTC_SGPU_BODY_TASK_SGPU_H_
#define BENCHAPPS_NBODY_UTC_SGPU_BODY_TASK_SGPU_H_

#include "Utc.h"
#include "UtcGpu.h"

using namespace iUtc;

template<typename T>
class BodySystemMGPU: public UserTaskBase{
private:
	unsigned int m_numBodies;
	T *m_pos;
	T *m_vel;

	T m_softeningSquared;
	T m_damping;
	T *oldPosBuffer;
	T *newPosBuffer;

	thread_local int local_numBodies;
	thread_local int local_startBodyIndex;

public:
	void initImpl(unsigned int numBodies,
			T softeningSquared,
			T damping,
			T *pos,
			T *vel);

	void runImpl(double** runtime,
			int loops,
			int outInterval,
			int blocksize,
			T deltaTime,
			T *outbuffer,
			MemType memtype);

};



#endif /* BENCHAPPS_NBODY_UTC_SGPU_BODY_TASK_SGPU_H_ */
