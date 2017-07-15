/*
 * bodysystem_kernel.h
 *
 *  Created on: Mar 8, 2017
 *      Author: chao
 */

#ifndef BODYSYSTEM_KERNEL_H_
#define BODYSYSTEM_KERNEL_H_

template<typename T>
struct vec3_t{
	typedef T Type;
};
template<>
struct vec3_t<float>{
	typedef float3 Type;
};
template<>
struct vec3_t<double>{
	typedef double3 Type;
};

template<typename T>
struct vec4_t{
	typedef T Type;
};
template<>
struct vec4_t<float>{
	typedef float4 Type;
};
template<>
struct vec4_t<double>{
	typedef double4 Type;
};

template <typename T>
__device__ void bodyBodyInteraction(
		typename vec3_t<T>::Type &accel,
		typename vec4_t<T>::Type &posMass0,
		T posMass1x,
		T posMass1y,
		T posMass1z,
		T posMass1w,
		T softeningSquared);

template <typename T>
__device__ typename vec3_t<T>::Type _computeNBodyForce_kernel(
			typename vec4_t<T>::Type &bodyPos,
			T* pos,
			int numTiles,
			T softeningSquared);

template <typename T>
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
		int start_bodyIndex);

template <typename T>
__device__ typename vec3_t<T>::Type _computeNBodyForceSmall_kernel(
		typename vec4_t<T>::Type &bodyPos,
		T* pos,
		int numTiles,
		int threadsperBody,
		T softeningSquared
		);

template <typename T>
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
		int start_bodyIndex);



#endif /* BODYSYSTEM_KERNEL_H_ */
