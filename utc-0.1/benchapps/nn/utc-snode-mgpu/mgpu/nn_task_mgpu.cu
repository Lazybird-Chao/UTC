/*
 * kmeans_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "nn_task_mgpu.h"

#include "../../../common/helper_err.h"
#include "nn_kernel.h"

template<typename T>
thread_local T* nnMGPU<T>::partial_obj_startPtr;
template<typename T>
thread_local int nnMGPU<T>::partial_numObjs;
template<typename T>
thread_local T* nnMGPU<T>::objsNN_array_startPtr;
template<typename T>
thread_local T* nnMGPU<T>::distance_array_startPtr;

template<typename T>
void nnMGPU<T>::initImpl(T*objects, T*objsNN, int numObjs, int numCoords, int numNN){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->objects = objects;
		this->objsNN = objsNN;
		this->numNN = numNN;
		this->numObjs = numObjs;
		this->numCoords = numCoords;
		objsNN_array = new T[numNN*numCoords*__numLocalThreads];
		distance_array = new T[numNN*__numLocalThreads];

	}
	intra_Barrier();
	objsNN_array_startPtr = objsNN_array + __localThreadId*numNN*numCoords;
	distance_array_startPtr = distance_array + __localThreadId*numNN;
	int objs_per_thread = numObjs/__numLocalThreads;
	if(__localThreadId < numObjs % __numLocalThreads){
		partial_numObjs = objs_per_thread+1;
		partial_obj_startPtr = objects + __localThreadId*(objs_per_thread+1)*numCoords;
	}
	else{
		partial_numObjs = objs_per_thread;
		partial_obj_startPtr = objects + (__localThreadId*objs_per_thread + numObjs % __numLocalThreads)*numCoords;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void nnMGPU<T>::runImpl(double runtime[][6], MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	/* target center */
	T *targetObj = new T[numCoords];
	for(int i=0; i<numCoords; i++)
		targetObj[i] = 0;


	GpuData<T> objs_d(partial_numObjs*numCoords, memtype);
	GpuData<T> distanceObjs(partial_numObjs, memtype);
	GpuData<T> targetObj_d(numCoords, memtype);
	objs_d.initH(partial_obj_startPtr);
	targetObj_d.initH(targetObj);

	timer0.start();
	timer.start();
	objs_d.sync();
	targetObj_d.sync();
	double copyinTime = timer.stop();

	double kernelTime1 =0;
	double kernelTime2 =0;
	double copyoutTime = 0;
	double hostCompTime = 0;

	int batchPerThread = 1;
	int blocksize = 256;
	int gridsize = (partial_numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	timer.start();
	distance_kernel<<<grid, block, 0, __streamId>>>(
			objs_d.getD(),
			numCoords,
			partial_numObjs,
			targetObj_d.getD(),
			distanceObjs.getD(true),
			batchPerThread);
	cudaStreamSynchronize(__streamId);
	checkCudaErr(cudaGetLastError());
	kernelTime1 = timer.stop();
	timer.start();

	GpuData<int> *topkIndexArray;
	int blocksize2 = 16;
	int gridsize2 = 1;
	if(partial_numObjs > blocksize2*numNN*8)
		gridsize2 = partial_numObjs/(blocksize2*numNN*8);
	//std::cout<<gridsize2<<std::endl;
	if(gridsize2 >=16){
		topkIndexArray  = new GpuData<int>(numNN*gridsize2, memtype);
		dim3 block2(blocksize2, 1, 1);
		dim3 grid2(gridsize2, 1, 1);
		topk_kernel<<<grid2, block2, 0, __streamId>>>(
				partial_numObjs,
				numNN,
				distanceObjs.getD(),
				topkIndexArray->getD(true));
		cudaStreamSynchronize(__streamId);
		checkCudaErr(cudaGetLastError());
	}
	kernelTime2 = timer.stop();

	timer.start();
	distanceObjs.sync();
	if(gridsize2 >=16){
		topkIndexArray->sync();
	}
	copyoutTime += timer.stop();


	/* find k nearest objs */
	if(gridsize2<16){
		T *distancePtr = distanceObjs.getH();
		timer.start();
		for(int i=0; i<numNN; i++){
			int min = 0;
			while(distancePtr[min]<0)
				min++;
			for(int j=0; j<partial_numObjs; j++){
				if(distancePtr[j]>=0 && distancePtr[min]>distancePtr[j])
					min = j;
			}
			distance_array_startPtr[i] = distancePtr[min];
			distancePtr[min] = -1;
			for(int j=0; j<numCoords; j++)
				objsNN_array_startPtr[i*numCoords + j] = partial_obj_startPtr[min*numCoords + j];

		}
		hostCompTime = timer.stop();
	}
	else{
		T *distancePtr = distanceObjs.getH();
		int *topkindexPtr = topkIndexArray->getH();
		timer.start();
		for(int i=0; i<numNN; i++){
			int min = 0;
			while(distancePtr[topkindexPtr[min]]<0)
				min++;
			for(int j=0; j<topkIndexArray->getSize(); j++){
				if(distancePtr[topkindexPtr[j]]>=0 &&
						distancePtr[topkindexPtr[min]]>distancePtr[topkindexPtr[j]])
					min = j;
			}
			distance_array_startPtr[i] = distancePtr[topkindexPtr[min]];
			distancePtr[topkindexPtr[min]] = -1;
			for(int j=0; j<numCoords; j++){
				objsNN_array_startPtr[i*numCoords +j] = partial_obj_startPtr[topkindexPtr[min]*numCoords+j];
			}
		}
		hostCompTime = timer.stop();
	}

	/* each thread finish finding k-nn
	 * then find global k-nn
	 */
	intra_Barrier();
	timer.start();
	if(getUniqueExecution()){
		for(int i=0; i<numNN; i++){
			int min = 0;
			while(distance_array[min]<0)
				min++;
			for(int j=0; j<numNN*__numLocalThreads; j++){
				if(distance_array[j]>=0 && distance_array[min]>distance_array[j])
					min = j;
			}
			distance_array[min] = -1;
			for(int j=0; j<numCoords; j++)
				objsNN[i*numCoords + j] = objsNN_array[min*numCoords + j];
		}
	}
	intra_Barrier();
	hostCompTime += timer.stop();
	totaltime = timer0.stop();


	//runtime[0] = kernelTime + copyinTime + copyoutTime + hostCompTime;
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1] = kernelTime1;
	runtime[__localThreadId][2] = kernelTime2;
	runtime[__localThreadId][3] = copyinTime;
	runtime[__localThreadId][4] = copyoutTime;
	runtime[__localThreadId][5] = hostCompTime;
	if(__localThreadId ==0){
		delete distance_array;
		delete objsNN_array;
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

template class nnMGPU<float>;
template class nnMGPU<double>;

