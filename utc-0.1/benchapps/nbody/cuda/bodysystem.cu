/*
 * bodysystem.cu
 *
 *  Created on: Mar 6, 2017
 *      Author: chao
 */

#include "bodysystem.h"
#include "bodysystem_kernel.h"
#include <cstring>

template <typename T>
BodySystem<T>::BodySystem(unsigned int numBodies)
    : m_numBodies(numBodies),
      m_bInitialized(false),
      m_softeningSquared(.00125f),
      m_damping(0.995f)
{
    m_pos = 0;
    m_vel = 0;
    m_pos_old_d =0;
    m_pos_new_d = 0;
    m_vel_d = 0;

    _initialize(numBodies);
}

template <typename T>
BodySystem<T>::~BodySystem()
{
    _finalize();
    m_numBodies = 0;
}

template <typename T>
void BodySystem<T>::_initialize(int numBodies)
{
    //assert(!m_bInitialized);

    m_numBodies = numBodies;

    m_pos    = new T[m_numBodies*4];
    m_vel    = new T[m_numBodies*4];

    memset(m_pos,   0, m_numBodies*4*sizeof(T));
    memset(m_vel,   0, m_numBodies*4*sizeof(T));


    m_bInitialized = true;
}

template <typename T>
void BodySystem<T>::_finalize()
{
    //assert(m_bInitialized);

    delete [] m_pos;
    delete [] m_vel;

    m_bInitialized = false;
}

template <typename T>
void BodySystem<T>::update(dim3 grid, dim3 block, T deltaTime)
{
    //assert(m_bInitialized);

	int ntiles = m_numBodies/block.x;
    _integrateNBodySystem_kernel<T><<<grid, block>>>(
    		m_pos_old_d,
    		m_pos_new_d,
    		m_vel_d,
    		m_numBodies,
    		deltaTime,
    		m_softeningSquared,
    		m_damping,
    		ntiles);

}

template<typename T>
void BodySystem<T>::update(dim3 grid, dim3 block, int threadsperBody, T deltaTime){
	int ntiles = m_numBodies/block.x;
	_integrateNBodySystemSmall_kernel<T><<<grid, block>>>(
	    		m_pos_old_d,
	    		m_pos_new_d,
	    		m_vel_d,
	    		m_numBodies,
	    		deltaTime,
	    		m_softeningSquared,
	    		m_damping,
	    		ntiles,
	    		threadsperBody);

}

template <typename T>
T *BodySystem<T>::getPosArray()
{
    return m_pos;
}

template <typename T>
void BodySystem<T>::setPosArray(T* pos)
{
	memcpy(m_pos, pos, m_numBodies*4*sizeof(T));
}

template <typename T>
T *BodySystem<T>::getDeviceOldPosArray()
{
    return m_pos_old_d;
}

template <typename T>
void BodySystem<T>::setDeviceOldPosArray(T* pos)
{
	//memcpy(m_pos, pos, m_numBodies*4*sizeof(T));
	m_pos_old_d = pos;
}

template <typename T>
T *BodySystem<T>::getDeviceNewPosArray()
{
    return m_pos_new_d;
}

template <typename T>
void BodySystem<T>::setDeviceNewPosArray(T* pos)
{
	//memcpy(m_pos, pos, m_numBodies*4*sizeof(T));
	m_pos_new_d = pos;
}



template <typename T>
T *BodySystem<T>::getVelArray()
{
    return m_vel;
}

template <typename T>
void BodySystem<T>::setVelArray(T* vel)
{
	memcpy(m_vel, vel, m_numBodies*4*sizeof(T));
}

template <typename T>
T *BodySystem<T>::getDeviceVelArray()
{
    return m_vel_d;
}

template <typename T>
void BodySystem<T>::setDeviceVelArray(T* vel)
{
	//memcpy(m_vel, vel, m_numBodies*4*sizeof(T));
	m_vel_d = vel;
}



template class BodySystem<float>;
template class BodySystem<double>;


