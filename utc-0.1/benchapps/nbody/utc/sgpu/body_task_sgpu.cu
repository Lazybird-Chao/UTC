/*
 * bodysystem.cu
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */

#include "body_task_sgpu.h"
#include "../../../common/helper_err.h"
#include "bodysystem_kernel.h"
#include <iostream>


template<typename T>
void BodySystemSGPU<T>::initImpl(unsigned int numBodies,
		T softeningSquared,
		T damping,
		T *pos,
		T *vel){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		m_numBodies = numBodies;
		m_softeningSquared = softeningSquared;
		m_damping = damping;
		m_pos = pos;
		m_vel = vel;
		//m_pos    = new T[m_numBodies*4];
		//m_vel    = new T[m_numBodies*4];

		//memcpy(m_pos, pos, m_numBodies*4*sizeof(T));
		//memcpy(m_vel, vel, m_numBodies*4*sizeof(T));
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void BodySystemSGPU<T>::runImpl(double* runtime,
			int loops,
			int outInterval,
			int blocksize,
			T deltaTime,
			T *outbuffer,
			MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer;

	GpuData<T> posBuffer[2] = {GpuData<T>(m_numBodies*4, memtype), GpuData<T>(m_numBodies*4, memtype)};
	GpuData<T> velBuffer(m_numBodies*4, memtype);
	posBuffer[0].initH(m_pos);
	velBuffer.initH(m_vel);

	/*
	 * copyin data
	 */
	timer.start();
	posBuffer[0].sync();
	velBuffer.sync();
	double copyinTime = timer.stop();

	/*
	 * iterate
	 */
	int mingridsize = 16;
	double kernelTime =0;
	double copyoutTime = 0;
	dim3 block(blocksize, 1,1);
	int threadsperBody = 1;
	if(m_numBodies/blocksize >= mingridsize)
		threadsperBody = 1;
	else
		threadsperBody = blocksize*mingridsize/m_numBodies;  //should keep this dividable
	dim3 grid(m_numBodies*threadsperBody/blocksize, 1,1);
	int ntiles = m_numBodies/block.x;
	int i=0;
	int posBufferIndex = 0;
	while(i<loops){
		timer.start();
		if(threadsperBody>1){
			_integrateNBodySystemSmall_kernel<T><<<grid, block>>>(
				    		posBuffer[posBufferIndex].getD(),
							posBuffer[1-posBufferIndex].getD(true),
				    		velBuffer.getD(true),
				    		m_numBodies,
				    		deltaTime,
				    		m_softeningSquared,
				    		m_damping,
				    		ntiles,
				    		threadsperBody);
		}
		else{
			_integrateNBodySystem_kernel<T><<<grid, block>>>(
						posBuffer[posBufferIndex].getD(),
						posBuffer[1-posBufferIndex].getD(true),
						velBuffer.getD(true),
			    		m_numBodies,
			    		deltaTime,
			    		m_softeningSquared,
			    		m_damping,
			    		ntiles);
		}
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaDeviceSynchronize());
		kernelTime += timer.stop();

		i++;
		if(i%outInterval ==0){
			/*
			 * copyout data
			 */
			timer.start();
			int offset = (i/outInterval -1)*m_numBodies*4;
			posBuffer[1-posBufferIndex].fetch(outbuffer+offset);
			copyoutTime += timer.stop();
		}
		posBufferIndex = 1-posBufferIndex;
	}
	runtime[0] = kernelTime + copyinTime + copyoutTime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

template class BodySystemSGPU<float>;
template class BodySystemSGPU<double>;

