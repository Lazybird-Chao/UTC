/*
 * bodysystem.h
 *
 *  Created on: Mar 5, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_
#define BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_

#include <cmath>

template<typename T>
class BodySystem{
public:
	BodySystem(unsigned int numBodies, int blockBodies, int bodyStartIndex);
	~BodySystem();

	void update(T deltaTime);

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

	T* getVelArray();
	void setVelArray(T* vel);

	T* getNewPosArray();

private:
	void _initialize(int numBodies);
	void _finalize();

	void _computeNBodyGravitation();
	void _integrateNBodySystem(T deltaTime);

	unsigned int m_numBodies;
	bool m_bInitialized;

	T *m_pos;
	T *m_vel;
	T *m_force;

	T m_softeningSquared;
	T m_damping;

	int m_blockBodies;
	int m_bodyStartIndex;
	T *m_newBlockPos;

};


inline double sqrt_T(double x)
{
    return sqrt(x);
}

inline float sqrt_T(float x)
{
    return sqrtf(x);
}

template <typename T>
void bodyBodyInteraction(T accel[3], T posMass0[4], T posMass1[4], T softeningSquared)
{
    T r[3];

    // r_01  [3 FLOPS]
    r[0] = posMass1[0] - posMass0[0];
    r[1] = posMass1[1] - posMass0[1];
    r[2] = posMass1[2] - posMass0[2];

    // d^2 + e^2 [6 FLOPS]
    T distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    //T invDist = (T)1.0 / (T)sqrt((double)distSqr);
    T invDist = 1.0/sqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = posMass1[3] * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}


#endif /* BENCHAPPS_NBODY_SEQ_BODYSYSTEM_H_ */
