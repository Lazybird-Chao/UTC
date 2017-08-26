/*
 * task.cc
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */
#include "task.h"
#include <algorithm>

template<typename T>
void NbodyInit<T>::runImpl(NBodyConfig config, T *pos, T *vel, float *color, float clusterScale,
                     float velocityScale, int numBodies, bool vec4vel){
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


template <typename T>
void Output<T>::runImpl(FILE** fp, T* buffer, float timestarmp, int numBodies){
	fprintf(*fp, "%f\n", timestarmp);
	for(int j=0; j<numBodies; j++){
		fprintf(*fp, "%.5f ", buffer[j*4 + 0]);
		fprintf(*fp, "%.5f ", buffer[j*4 +1]);
		fprintf(*fp, "%.5f ", buffer[j*4 +2]);
		fprintf(*fp, "%.5f\n", buffer[j*4 +3]);
	}
}

template class NbodyInit<float>;
template class NbodyInit<double>;
template class Output<float>;
template class Output<double>;






