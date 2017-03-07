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
struct vec3_t{
	typename T Type;
};
template<>
struct vec3_t<float>{
	typename float3 Type;
};
template<>
struct vec3_t<double>{
	typename double3 Type;
};

template<typename T>
struct vec4_t{
	typename T Type;
};
template<>
struct vec4_t<float>{
	typename float4 Type;
};
template<>
struct vec4_t<double>{
	typename double4 Type;
};

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

	void setDeviceForceArray(T* force);

private:
	void _initialize(int numBodies);
	void _finalize();

	__device__ typename vec3_t<T>::Type _computeNBodyForce(
			typename vec4_t<T> &bodyPos,
			T* pos,
			int numTiles,
			T softeningSquared);
	__global__ void _integrateNBodySystem(
			T* pos_old,
			T* pos_new,
			T* vel,
			int numBodies,
			T deltaTime,
			T softeningSquared,
			T damping,
			int numTiles);

	__device__ typename vec3_t<T>::Type _computeNBodyForceSmall(
			typename vec4_t<T> &bodyPos,
			T* pos,
			int numTiles,
			int threadsperBody,
			T softeningSquared
			)

	__global__ void _integrateNBodySystemSmall(
			T* pos_old,
			T* pos_new,
			T* vel,
			int numBodies,
			T deltaTime,
			T softeningSquared,
			T damping,
			int numTiles,
			int threadsperBody);

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
