/*
 * nbody_kernel.cu
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */
#include "nbody_kernel.h"

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
    r[0] = posMass1x - posMass0.x;
    r[1] = posMass1y - posMass0.y;
    r[2] = posMass1z - posMass0.z;

    // d^2 + e^2 [6 FLOPS]
    T distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    //T invDist = (T)1.0 / (T)sqrt((double)distSqr);
    T invDist = 1.0/sqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = posMass1w * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel.x += r[0] * s;
    accel.y += r[1] * s;
    accel.z += r[2] * s;
    /*if(threadIdx.x==1 && blockIdx.x ==0)
		printf("%f %f %f\n", accel.x, accel.y, accel.z);*/
}



template <typename T>
__device__ typename vec3_t<T>::Type _computeNBodyForce_kernel(
		typename vec4_t<T>::Type &bodyPos,
		T* pos,
		int numTiles,
		T softeningSquared
		){
	__shared__ T sharedPos[4][512];  //assume block size max is 512
	typename vec3_t<T>::Type acc = {0, 0, 0};
	for(int tile=0; tile<numTiles; tile++){
		sharedPos[0][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+0];
		sharedPos[1][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+1];
		sharedPos[2][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+2];
		sharedPos[3][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+3];
		__syncthreads();

		for(int i=0; i<blockDim.x; i++){
			bodyBodyInteraction<T>(acc, bodyPos, sharedPos[0][i],
					sharedPos[1][i], sharedPos[2][i], sharedPos[3][i], softeningSquared);
			/*if(threadIdx.x==1 && blockIdx.x==0)
				printf("%f %f %f\n", acc.x, acc.y, acc.z);*/
		}
		__syncthreads();
	}
	return acc;
}

template<typename T>
__global__ void _integrateNBodySystem_kernel(
		T* pos_old,
		T* pos_new,
		T* vel,
		int numBodies,
		T deltaTime,
		T softeningSquared,
		T damping,
		int numTiles,
		int local_numBodies,
		int start_bodyIndex){
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx>=local_numBodies)
		return;
	typename vec4_t<T>::Type bodyPos;
	bodyPos.x = pos_old[(start_bodyIndex+idx)*4+0];
	bodyPos.y = pos_old[(start_bodyIndex+idx)*4+1];
	bodyPos.z = pos_old[(start_bodyIndex+idx)*4+2];
	bodyPos.w = pos_old[(start_bodyIndex+idx)*4+3];
	T invMass = bodyPos.w;
	typename vec3_t<T>::Type acc = _computeNBodyForce_kernel(
			bodyPos, pos_old, numTiles, softeningSquared);

	typename vec4_t<T>::Type curvel;
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
	pos_new[idx*4+3] = bodyPos.w;

	vel[idx*4+0] = curvel.x;
	vel[idx*4+1] = curvel.y;
	vel[idx*4+2] = curvel.z;
	vel[idx*4+3] = curvel.w;

}

template <typename T>
__device__ typename vec3_t<T>::Type _computeNBodyForceSmall_kernel(
		typename vec4_t<T>::Type &bodyPos,
		T* pos,
		int numTiles,
		int threadsperBody,
		T softeningSquared
		){
	__shared__ T sharedPos[4][512];  //assume block size max is 512

	typename vec3_t<T>::Type acc = {0, 0, 0};
	__shared__ typename vec3_t<T>::Type acc_gather[512];
	acc_gather[threadIdx.x]=acc;
	int offset = threadIdx.x/(blockDim.x/threadsperBody);
	offset *= (blockDim.x/threadsperBody);
	for(int tile=0; tile<numTiles; tile++){
		sharedPos[0][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+0];
		sharedPos[1][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+1];
		sharedPos[2][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+2];
		sharedPos[3][threadIdx.x] = pos[4*blockDim.x*tile+4*threadIdx.x+3];
		__syncthreads();

		for(int i=0; i<blockDim.x/threadsperBody; i++){
			bodyBodyInteraction<T>(acc, bodyPos, sharedPos[0][i+offset],
					sharedPos[1][i+offset], sharedPos[2][i+offset], sharedPos[3][i+offset], softeningSquared);
		}
		__syncthreads();
	}
	acc_gather[threadIdx.x].x += acc.x;
	acc_gather[threadIdx.x].y += acc.y;
	acc_gather[threadIdx.x].z += acc.z;
	__syncthreads();
	if(threadIdx.x<blockDim.x/threadsperBody){
		acc = acc_gather[threadIdx.x];
		for(int i=1; i<threadsperBody; i++){
			acc.x += acc_gather[threadIdx.x+i*(blockDim.x/threadsperBody)].x;
			acc.y += acc_gather[threadIdx.x+i*(blockDim.x/threadsperBody)].y;
			acc.z += acc_gather[threadIdx.x+i*(blockDim.x/threadsperBody)].z;
		}
		/*if(threadIdx.x==1 && blockIdx.x==0)
			printf("%f %f %f\n", acc.x, acc.y, acc.z);*/
	}
	__syncthreads();
	return acc;
}

template<typename T>
__global__ void _integrateNBodySystemSmall_kernel(
		T* pos_old,
		T* pos_new,
		T* vel,
		int numBodies,
		T deltaTime,
		T softeningSquared,
		T damping,
		int numTiles,
		int threadsperBody,
		int local_numBodies,
		int start_bodyIndex){
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx>=local_numBodies*threadsperBody)
		return;
	typename vec4_t<T>::Type bodyPos;
	int blockoffset = blockIdx.x*blockDim.x/threadsperBody;
	int threadoffset = threadIdx.x%(blockDim.x/threadsperBody);
	int offset = (blockoffset + threadoffset)*4;
	bodyPos.x = pos_old[start_bodyIndex*4 + offset+0];
	bodyPos.y = pos_old[start_bodyIndex*4 + offset+1];
	bodyPos.z = pos_old[start_bodyIndex*4 + offset+2];
	bodyPos.w = pos_old[start_bodyIndex*4 + offset+3];
	T invMass = bodyPos.w;
	typename vec3_t<T>::Type acc = _computeNBodyForceSmall_kernel(
			bodyPos, pos_old, numTiles, threadsperBody, softeningSquared);

	if(threadIdx.x<blockDim.x/threadsperBody){
		typename vec4_t<T>::Type curvel;
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
		pos_new[offset+3] = bodyPos.w;

		vel[offset+0] = curvel.x;
		vel[offset+1] = curvel.y;
		vel[offset+2] = curvel.z;
		vel[offset+3] = curvel.w;
	}

}


template __device__ void bodyBodyInteraction(typename vec3_t<float>::Type &accel,
		typename vec4_t<float>::Type &posMass0,
		float posMass1x,
		float posMass1y,
		float posMass1z,
		float posMass1w,
		float softeningSquared);
template __device__ void bodyBodyInteraction(typename vec3_t<double>::Type &accel,
		typename vec4_t<double>::Type &posMass0,
		double posMass1x,
		double posMass1y,
		double posMass1z,
		double posMass1w,
		double softeningSquared);

template __device__ typename vec3_t<float>::Type _computeNBodyForce_kernel(
			typename vec4_t<float>::Type &bodyPos,
			float* pos,
			int numTiles,
			float softeningSquared);
template __device__ typename vec3_t<double>::Type _computeNBodyForce_kernel(
			typename vec4_t<double>::Type &bodyPos,
			double* pos,
			int numTiles,
			double softeningSquared);

template __global__ void _integrateNBodySystem_kernel(
		float* pos_old,
		float* pos_new,
		float* vel,
		int numBodies,
		float deltaTime,
		float softeningSquared,
		float damping,
		int numTiles,
		int local_numBodies,
		int start_bodyIndex);
template __global__ void _integrateNBodySystem_kernel(
		double* pos_old,
		double* pos_new,
		double* vel,
		int numBodies,
		double deltaTime,
		double softeningSquared,
		double damping,
		int numTiles,
		int local_numBodies,
		int start_bodyIndex);

template __device__ typename vec3_t<float>::Type _computeNBodyForceSmall_kernel(
		typename vec4_t<float>::Type &bodyPos,
		float* pos,
		int numTiles,
		int threadsperBody,
		float softeningSquared
		);
template __device__ typename vec3_t<double>::Type _computeNBodyForceSmall_kernel(
		typename vec4_t<double>::Type &bodyPos,
		double* pos,
		int numTiles,
		int threadsperBody,
		double softeningSquared
		);

template __global__ void _integrateNBodySystemSmall_kernel(
		float* pos_old,
		float* pos_new,
		float* vel,
		int numBodies,
		float deltaTime,
		float softeningSquared,
		float damping,
		int numTiles,
		int threadsperBody,
		int local_numBodies,
		int start_bodyIndex);
template __global__ void _integrateNBodySystemSmall_kernel(
		double* pos_old,
		double* pos_new,
		double* vel,
		int numBodies,
		double deltaTime,
		double softeningSquared,
		double damping,
		int numTiles,
		int threadsperBody,
		int local_numBodies,
		int start_bodyIndex);




