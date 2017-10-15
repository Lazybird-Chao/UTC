/*
 * task.h
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_UTC_CPU_TASK_H_
#define BENCHAPPS_NBODY_UTC_CPU_TASK_H_
#include <cstdio>
#include <cmath>
#include "Utc.h"

enum NBodyConfig
{
    NBODY_CONFIG_RANDOM,
    NBODY_CONFIG_SHELL,
    NBODY_CONFIG_EXPAND,
    NBODY_NUM_CONFIGS
};

template <typename T>
struct vec3{
	T x;
	T y;
	T z;
};

template <typename T>
struct vec4{
	T x;
	T y;
	T z;
	T w;
};

template struct vec3<float>;
template struct vec3<double>;
template struct vec4<float>;
template struct vec4<double>;

inline vec3<float>
scalevec(vec3<float> &vector, float scalar)
{
	vec3<float> rt = vector;
    rt.x *= scalar;
    rt.y *= scalar;
    rt.z *= scalar;
    return rt;
}

inline float
normalize(vec3<float> &vector)
{
	 float dist = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);

    if (dist > 1e-6)
    {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}


inline float
dot(vec3<float> v0, vec3<float> v1)
{
    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
}


inline vec3<float>
cross(vec3<float> v0, vec3<float> v1)
{
	vec3<float> rt;
    rt.x = v0.y*v1.z-v0.z*v1.y;
    rt.y = v0.z*v1.x-v0.x*v1.z;
    rt.z = v0.x*v1.y-v0.y*v1.x;
    return rt;
}

template<typename T>
class NbodyInit:public UserTaskBase{
public:
	void runImpl(NBodyConfig config, T *pos, T *vel, float *color, float clusterScale,
                     float velocityScale, int numBodies, bool vec4vel);
};


template<typename T>
class Output:public UserTaskBase{
public:
	void runImpl(FILE** fp, T* buffer, float timestamp, int numBodies );
};




#endif /* BENCHAPPS_NBODY_UTC_CPU_TASK_H_ */
