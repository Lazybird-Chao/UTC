/*
 * bodysystem.cu
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */

#include "body_task_mgpu.h"
#include "../../../common/helper_err.h"
#include "bodysystem_kernel.h"
#include <iostream>

template<typename T>
thread_local int BodySystemMGPU<T>::local_numBodies;
template<typename T>
thread_local int BodySystemMGPU<T>::local_startBodyIndex;

template<typename T>
void BodySystemMGPU<T>::initImpl(unsigned int numBodies,
		T softeningSquared,
		T damping,
		T *pos,
		T *vel){
	if(__localThreadId ==0){
		//std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		m_numBodies = numBodies;
		m_softeningSquared = softeningSquared;
		m_damping = damping;
		m_pos = pos;
		m_vel = vel;
		oldPosBuffer = new T[m_numBodies*4];
		newPosBuffer = new T[m_numBodies*4];
		memcpy(oldPosBuffer, m_pos, m_numBodies*4*sizeof(T));
		//m_pos    = new T[m_numBodies*4];
		//m_vel    = new T[m_numBodies*4];

		//memcpy(m_pos, pos, m_numBodies*4*sizeof(T));
		//memcpy(m_vel, vel, m_numBodies*4*sizeof(T));
	}
	intra_Barrier();
	int bodiesPerThread = numBodies/__numLocalThreads;
	if(__localThreadId < numBodies % __numLocalThreads){
		local_numBodies = bodiesPerThread +1;
		local_startBodyIndex = __localThreadId *(bodiesPerThread +1);
	}
	else{
		local_numBodies = bodiesPerThread;
		local_startBodyIndex = __localThreadId * bodiesPerThread + numBodies%__numLocalThreads;
	}

	intra_Barrier();
	/*if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}*/
}

template<typename T>
void BodySystemMGPU<T>::runImpl(double runtime[][4],
			int loops,
			int outInterval,
			int blocksize,
			T deltaTime,
			T *outbuffer,
			MemType memtype){
	/*if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}*/
	Timer timer, timer0;
	double totaltime;

	//timer0.start();
	GpuData<T> posBuffer[2] = {GpuData<T>(m_numBodies*4, memtype), GpuData<T>(local_numBodies*4, memtype)};
	GpuData<T> velBuffer(local_numBodies*4, memtype);
	//posBuffer[0].initH(oldPosBuffer);
	velBuffer.initH(m_vel+local_startBodyIndex*4);

	/*
	 * copyin data
	 */
	timer0.start();
	timer.start();
	//posBuffer[0].sync();
	velBuffer.sync();
	double copyinTime = timer.stop();

	/*
	 * iterate
	 */
	int mingridsize = 1;
	double kernelTime =0;
	double copyoutTime = 0;
	dim3 block(blocksize, 1,1);
	int threadsperBody = 1;
	if(local_numBodies/blocksize >= mingridsize)
		threadsperBody = 1;
	else
		threadsperBody = blocksize*mingridsize/local_numBodies;  //should keep this dividable
	dim3 grid((local_numBodies*threadsperBody+blocksize-1)/blocksize, 1,1);
	int ntiles = m_numBodies/block.x;
	int i=0;
	//int posBufferIndex = 0;
	T* tmp[2];
	tmp[0] = oldPosBuffer;
	tmp[1] = newPosBuffer;
	int cur = 0;
	while(i<loops){
		//copy all bodies new pos data to gpu
		timer.start();
		posBuffer[0].putD(tmp[cur]);
		copyinTime += timer.stop();

		//compute new pos
		timer.start();
		if(threadsperBody>1){
			_integrateNBodySystemSmall_kernel<T><<<grid, block, 0, __streamId>>>(
				    		posBuffer[0].getD(),
							posBuffer[1].getD(true),
				    		velBuffer.getD(true),
				    		m_numBodies,
				    		deltaTime,
				    		m_softeningSquared,
				    		m_damping,
				    		ntiles,
				    		threadsperBody,
				    		local_numBodies,
				    		local_startBodyIndex);
		}
		else{
			_integrateNBodySystem_kernel<T><<<grid, block, 0, __streamId>>>(
						posBuffer[0].getD(),
						posBuffer[1].getD(true),
						velBuffer.getD(true),
			    		m_numBodies,
			    		deltaTime,
			    		m_softeningSquared,
			    		m_damping,
			    		ntiles,
			    		local_numBodies,
			    		local_startBodyIndex);
		}
		checkCudaErr(cudaGetLastError());
		//checkCudaErr(cudaDeviceSynchronize());
		checkCudaErr(cudaStreamSynchronize(__streamId));
		kernelTime += timer.stop();

		/*
		 * each thread update part of all bodies,
		 * gather results of each thread together
		 */
		timer.start();
		posBuffer[1].fetchD(tmp[1-cur]+local_startBodyIndex*4);
		copyoutTime += timer.stop();

		intra_Barrier();
		i++;
		if(i%outInterval ==0){
			int offset = (i/outInterval -1)*m_numBodies*4;
			memcpy(outbuffer+offset+local_startBodyIndex*4, tmp[1-cur]+local_startBodyIndex*4,
					local_numBodies*4*sizeof(T));
		}
		cur = 1-cur;
		//posBufferIndex = 1-posBufferIndex;
	}
	runtime[__localThreadId][0] = timer0.stop();
	//runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[__localThreadId][1] = kernelTime;
	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;

	if(__localThreadId ==0){
		delete oldPosBuffer;
		delete newPosBuffer;

		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

template class BodySystemMGPU<float>;
template class BodySystemMGPU<double>;

