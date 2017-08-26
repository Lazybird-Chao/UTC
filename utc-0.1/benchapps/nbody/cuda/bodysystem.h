/*
 * bodysystem.h
 *
 *  Created on: Mar 5, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_
#define BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_

#include <cuda_runtime.h>



template<typename T>
class BodySystem{
public:
	BodySystem(unsigned int numBodies);
	~BodySystem();

	void update(dim3 grid, dim3 block, T deltaTime);

	void update(dim3 grid, dim3 block, int threadsperBody, T deltaTime);

	void setSoftening(T softening){
		m_softeningSquared = softening * softening;
	}

	void setDamping(T damping)
	{
		m_damping = damping;
	}

	unsigned int getNumBodies() const
	{
		return m_numBodies;
	}

	T* getPosArray();
	void setPosArray(T* pos);

	T* getDeviceOldPosArray();
	void setDeviceOldPosArray(T* pos);

	T* getDeviceNewPosArray();
	void setDeviceNewPosArray(T* pos);

	T* getVelArray();
	void setVelArray(T* vel);

	T* getDeviceVelArray();
	void setDeviceVelArray(T* vel);


private:
	void _initialize(int numBodies);
	void _finalize();


	unsigned int m_numBodies;
	bool m_bInitialized;

	T *m_pos;
	T *m_vel;

	T *m_pos_old_d;
	T *m_pos_new_d;
	T *m_vel_d;

	T m_softeningSquared;
	T m_damping;
};



#endif /* BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_ */
