/*
 * nbody.h
 *
 *  Created on: Mar 5, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_NBODY_SEQ_NBODY_H_
#define BENCHAPPS_NBODY_SEQ_NBODY_H_

#include <algorithm>
#include <cmath>

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


// utility function
template <typename T>
void randomizeBodies(NBodyConfig config, T *pos, T *vel, float *color, float clusterScale,
                     float velocityScale, int numBodies, bool vec4vel)
{
    switch (config)
    {
        default:
        case NBODY_CONFIG_RANDOM:
            {
                float scale = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
                float vscale = velocityScale * scale;

                int p = 0, v = 0;
                int i = 0;

                while (i < numBodies)
                {
                    vec3<float> point;
                    //const int scale = 16;
                    point.x = rand() / (float) RAND_MAX * 2 - 1;
                    point.y = rand() / (float) RAND_MAX * 2 - 1;
                    point.z = rand() / (float) RAND_MAX * 2 - 1;
                    float lenSqr = dot(point, point);

                    if (lenSqr > 1)
                        continue;

                    vec3<float> velocity;
                    velocity.x = rand() / (float) RAND_MAX * 2 - 1;
                    velocity.y = rand() / (float) RAND_MAX * 2 - 1;
                    velocity.z = rand() / (float) RAND_MAX * 2 - 1;
                    lenSqr = dot(velocity, velocity);

                    if (lenSqr > 1)
                        continue;

                    pos[p++] = point.x * scale; // pos.x
                    pos[p++] = point.y * scale; // pos.y
                    pos[p++] = point.z * scale; // pos.z
                    pos[p++] = 1.0f; // mass

                    vel[v++] = velocity.x * vscale; // pos.x
                    vel[v++] = velocity.y * vscale; // pos.x
                    vel[v++] = velocity.z * vscale; // pos.x

                    if (vec4vel) vel[v++] = 1.0f; // inverse mass

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_SHELL:
            {
                float scale = clusterScale;
                float vscale = scale * velocityScale;
                float inner = 2.5f * scale;
                float outer = 4.0f * scale;

                int p = 0, v=0;
                int i = 0;

                while (i < numBodies)//for(int i=0; i < numBodies; i++)
                {
                    float x, y, z;
                    x = rand() / (float) RAND_MAX * 2 - 1;
                    y = rand() / (float) RAND_MAX * 2 - 1;
                    z = rand() / (float) RAND_MAX * 2 - 1;

                    vec3<float> point = {x, y, z};
                    float len = normalize(point);

                    if (len > 1)
                        continue;

                    pos[p++] =  point.x * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                    pos[p++] =  point.y * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                    pos[p++] =  point.z * (inner + (outer - inner) * rand() / (float) RAND_MAX);
                    pos[p++] = 1.0f;

                    x = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                    y = 0.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                    z = 1.0f; // * (rand() / (float) RAND_MAX * 2 - 1);
                    vec3<float> axis = {x, y, z};
                    normalize(axis);

                    if (1 - dot(point, axis) < 1e-6)
                    {
                        axis.x = point.y;
                        axis.y = point.x;
                        normalize(axis);
                    }

                    //if (point.y < 0) axis = scalevec(axis, -1);
                    vec3<float> vv = {(float)pos[4*i], (float)pos[4*i+1], (float)pos[4*i+2]};
                    vv = cross(vv, axis);
                    vel[v++] = vv.x * vscale;
                    vel[v++] = vv.y * vscale;
                    vel[v++] = vv.z * vscale;

                    if (vec4vel) vel[v++] = 1.0f;

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_EXPAND:
            {
                float scale = clusterScale * numBodies / (1024.f);

                if (scale < 1.0f)
                    scale = clusterScale;

                float vscale = scale * velocityScale;

                int p = 0, v = 0;

                for (int i=0; i < numBodies;)
                {
                	vec3<float> point;

                    point.x = rand() / (float) RAND_MAX * 2 - 1;
                    point.y = rand() / (float) RAND_MAX * 2 - 1;
                    point.z = rand() / (float) RAND_MAX * 2 - 1;

                    float lenSqr = dot(point, point);

                    if (lenSqr > 1)
                        continue;

                    pos[p++] = point.x * scale; // pos.x
                    pos[p++] = point.y * scale; // pos.y
                    pos[p++] = point.z * scale; // pos.z
                    pos[p++] = 1.0f; // mass
                    vel[v++] = point.x * vscale; // pos.x
                    vel[v++] = point.y * vscale; // pos.x
                    vel[v++] = point.z * vscale; // pos.x

                    if (vec4vel) vel[v++] = 1.0f; // inverse mass

                    i++;
                }
            }
            break;
    }

    if (color)
    {
        int v = 0;

        for (int i=0; i < numBodies; i++)
        {
            //const int scale = 16;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = rand() / (float) RAND_MAX;
            color[v++] = 1.0f;
        }
    }

}



#endif /* BENCHAPPS_NBODY_SEQ_NBODY_H_ */
