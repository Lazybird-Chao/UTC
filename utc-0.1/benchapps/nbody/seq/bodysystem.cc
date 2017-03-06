/*
 * bodysystem.cc
 *
 *  Created on: Mar 5, 2017
 *      Author: Chao
 */

#include "bodysystem.h"
#include "nbody.h"
#include <cstring>

template <typename T>
BodySystem<T>::BodySystem(unsigned int numBodies)
    : m_numBodies(numBodies),
      m_bInitialized(false),
      m_force(0),
      m_softeningSquared(.00125f),
      m_damping(0.995f)
{
    m_pos = 0;
    m_vel = 0;

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
    m_force  = new T[m_numBodies*3];

    memset(m_pos,   0, m_numBodies*4*sizeof(T));
    memset(m_vel,   0, m_numBodies*4*sizeof(T));
    memset(m_force, 0, m_numBodies*3*sizeof(T));

    m_bInitialized = true;
}

template <typename T>
void BodySystem<T>::_finalize()
{
    //assert(m_bInitialized);

    delete [] m_pos;
    delete [] m_vel;
    delete [] m_force;

    m_bInitialized = false;
}

template <typename T>
void BodySystem<T>::update(T deltaTime)
{
    //assert(m_bInitialized);

	_computeNBodyGravitation();

    _integrateNBodySystem(deltaTime);

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
void BodySystem<T>::_computeNBodyGravitation()
{
    for (int i = 0; i < m_numBodies; i++)
    {
        int indexForce = 3*i;

        T acc[3] = {0, 0, 0};

        // We unroll this loop 4X for a small performance boost.
        int j = 0;

        while (j < m_numBodies)
        {
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
        }

        m_force[indexForce  ] = acc[0];
        m_force[indexForce+1] = acc[1];
        m_force[indexForce+2] = acc[2];
    }
}

template <typename T>
void BodySystem<T>::_integrateNBodySystem(T deltaTime)
{

    for (int i = 0; i < m_numBodies; ++i)
    {
        int index = 4*i;
        int indexForce = 3*i;


        T pos[3], vel[3], force[3];
        pos[0] = m_pos[index+0];
        pos[1] = m_pos[index+1];
        pos[2] = m_pos[index+2];
        T invMass = m_pos[index+3];

        vel[0] = m_vel[index+0];
        vel[1] = m_vel[index+1];
        vel[2] = m_vel[index+2];

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

        m_pos[index+0] = pos[0];
        m_pos[index+1] = pos[1];
        m_pos[index+2] = pos[2];

        m_vel[index+0] = vel[0];
        m_vel[index+1] = vel[1];
        m_vel[index+2] = vel[2];
    }
}

template class BodySystem<float>;
template class BodySystem<double>;






