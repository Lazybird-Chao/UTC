/*
 * bodysystem_kernel.cc
 *
 *  Created on: Mar 6, 2017
 *      Author: chao
 */

#include "bodysystem.h"

__device__ inline double sqrt_T(double x)
{
    return sqrt(x);
}

__device__ inline float sqrt_T(float x)
{
    return sqrtf(x);
}

template <typename T>
__device__ void bodyBodyInteraction(
		typename vec3_t<T>::Type &accel,
		typename vec4_t<T>::Type &posMass0,
		T posMass1x,
		T posMass1y,
		T posMass1z,
		T posMass1w,
		T softeningSquared)
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
    accel.x += r[0] * s;
    accel.y += r[1] * s;
    accel.z += r[2] * s;
}



template <typename T>
__device__ typename vec3_t<T>::Type BodySystem<T>::_computeNBodyForce(
		typename vec4_t<T> &bodyPos,
		T* pos,
		int numTiles,
		T softeningSquared
		){
	__shared__ T sharedPos[4][512];  //assume block size max is 512
	typename vec3_t<T> acc = {0, 0, 0};
	for(int tile=0; tile<numTiles; tile++){
		sharedPos[0][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+0];
		sharedPos[1][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+1];
		sharedPos[2][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+2];
		sharedPos[3][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+3];
		__syncthreads();

		for(int i=0; i<blockDim.x; i++){
			bodyBodyInteraction<T>(acc, bodyPos, sharedPos[0][i],
					sharedPos[1][i], sharedPos[2][i], sharedPos[3][i], softeningSquared);
		}
		__syncthreads();
	}
	return acc;
}

template<typename T>
__global__ void BodySystem<T>::_integrateNBodySystem(
		T* pos_old,
		T* pos_new,
		T* vel,
		int numBodies,
		T deltaTime,
		T softeningSquared,
		T damping,
		int numTiles){
	int idx = blockIdx.x *blockDim.x + ThreadIdx.x;
	if(idx>=numBodies)
		return;
	typename vec4_t<T> bodyPos;
	bodyPos.x = pos_old[idx*4+0];
	bodyPos.y = pos_old[idx*4+1];
	bodyPos.z = pos_old[idx*4+2];
	bodyPos.w = pos_old[idx*4+3];
	T invMass = bodyPos.w;
	typename vec3_t<T> acc = _computeNBodyForce(bodyPos, pos_old, numTiles, softeningSquared);

	typename vec4_t<T> curvel;
	curvel.x = vel[idx*4+0];
	curvel.y = vel[idx*4+1];
	curvel.z = vel[idx*4+2];
	curvel.w = vel[idx*4+3];

	curvel.x += (acc.x * invMass) * deltaTime;
	curvel.y += (acc.y * invMass) * deltaTime;
	curvel.z += (acc.z * invMass) * deltaTime;

	curvel.x *= damping;
	curvel.y *= damping;
	curvel.z *= damping;

	// new position = old position + velocity * deltaTime
	bodyPos.x += curvel.x * deltaTime;
	bodyPos.y += curvel.y * deltaTime;
	bodyPos.z += curvel.z * deltaTime;

	pos_new[idx*4+0] = bodyPos.x;
	pos_new[idx*4+1] = bodyPos.y;
	pos_new[idx*4+2] = bodyPos.z;
	pos_new[idx*4+4] = bodyPos.w;

	vel[idx*4+0] = curvel.x;
	vel[idx*4+1] = curvel.y;
	vel[idx*4+2] = curvel.z;
	vel[idx*4+3] = curvel.w;

}

template <typename T>
__device__ typename vec3_t<T>::Type BodySystem<T>::_computeNBodyForceSmall(
		typename vec4_t<T> &bodyPos,
		T* pos,
		int numTiles,
		int threadsperBody,
		T softeningSquared
		){
	__shared__ T sharedPos[4][512];  //assume block size max is 512

	typename vec3_t<T> acc = {0, 0, 0};
	__shared__ typename vec3_t<T> acc_gather[512];
	acc_gather[threadIdx.x]=acc;
	int offset = threadIdx.x/(blockDim.x/threadsperBody);
	offset *= (blockDim.x/threadsperBody);
	for(int tile=0; tile<numTiles; tile++){
		sharedPos[0][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+0];
		sharedPos[1][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+1];
		sharedPos[2][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+2];
		sharedPos[3][threadIdx.x] = pos[blockDim.x*tile+threadIdx.x+3];
		__syncthreads();

		for(int i=0; i<blockDim.x/threadsperBody; i++){
			bodyBodyInteraction<T>(acc, bodyPos, sharedPos[0][i+offset],
					sharedPos[1][i+offset], sharedPos[2][i+offset], sharedPos[3][i+offset], softeningSquared);
		}
		acc_gather[threadIdx.x] += acc;
		__syncthreads();
	}
	if(threadIdx.x<blockDim.x/threadsperBody){
		acc = acc_gather[threadIdx.x];
		for(int i=1; i<threadsperBody; i++)
			acc += acc_gather[threadIdx.x+i*(blockDim.x/threadsperBody)];
	}
	__syncthreads();
	return acc;
}

template<typename T>
__global__ void BodySystem<T>::_integrateNBodySystemSmall(
		T* pos_old,
		T* pos_new,
		T* vel,
		int numBodies,
		T deltaTime,
		T softeningSquared,
		T damping,
		int numTiles,
		int threadsperBody){
	int idx = blockIdx.x *blockDim.x + ThreadIdx.x;
	if(idx>=numBodies*threadsperBody)
		return;
	typename vec4_t<T> bodyPos;
	int blockoffset = blockIdx.x*blockDim.x/threadsperBody;
	int threadoffset = threadIdx.x%(blockDim.x/threadsperBody);
	int offset = (blockoffset + threadoffset)*4;
	bodyPos.x = pos_old[offset+0];
	bodyPos.y = pos_old[offset+1];
	bodyPos.z = pos_old[offset+2];
	bodyPos.w = pos_old[offset+3];
	T invMass = bodyPos.w;
	typename vec3_t<T> acc = _computeNBodyForce(bodyPos, pos_old, numTiles, softeningSquared);

	if(threadIdx.x<blockDim.x/threadsperBody){
		typename vec4_t<T> curvel;
		curvel.x = vel[offset+0];
		curvel.y = vel[offset+1];
		curvel.z = vel[offset+2];
		curvel.w = vel[offset+3];

		curvel.x += (acc.x * invMass) * deltaTime;
		curvel.y += (acc.y * invMass) * deltaTime;
		curvel.z += (acc.z * invMass) * deltaTime;

		curvel.x *= damping;
		curvel.y *= damping;
		curvel.z *= damping;

		// new position = old position + velocity * deltaTime
		bodyPos.x += curvel.x * deltaTime;
		bodyPos.y += curvel.y * deltaTime;
		bodyPos.z += curvel.z * deltaTime;

		pos_new[offset+0] = bodyPos.x;
		pos_new[offset+1] = bodyPos.y;
		pos_new[offset+2] = bodyPos.z;
		pos_new[offset+4] = bodyPos.w;

		vel[offset+0] = curvel.x;
		vel[offset+1] = curvel.y;
		vel[offset+2] = curvel.z;
		vel[offset+3] = curvel.w;
	}

}
