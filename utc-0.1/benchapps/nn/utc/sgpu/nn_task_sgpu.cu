/*
 * kmeans_task_sgpu.cu
 *
 *  Created on: Mar 22, 2017
 *      Author: chao
 */

#include "nn_task_sgpu.h"

#include "../../../common/helper_err.h"
#include "nn_kernel.h"



template<typename T>
void nnSGPU<T>::initImpl(T*objects, T*objsNN, int numObjs, int numCoords, int numNN){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";

		this->objects = objects;
		this->objsNN = objsNN;
		this->numNN = numNN;
		this->numObjs = numObjs;
		this->numCoords = numCoords;

	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
void nnSGPU<T>::runImpl(double *runtime, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	/* target center */
	float *targetObj = new float[numCoords];
	for(int i=0; i<numCoords; i++)
		targetObj[i] = 0;


	GpuData<T> objs_d(numObjs*numCoords);
	GpuData<T> distanceObjs(numObjs);
	GpuData<T> targetObj_d(numCoords);
	objs_d.initH(objects);
	targetObj_d.initH(targetObj);

	timer0.start();
	timer.start();
	objs_d.sync();
	targetObj_d.sync();
	double copyinTime = timer.stop();

	double kernelTime =0;
	double copyoutTime = 0;
	double hostCompTime = 0;

	int batchPerThread = 1;
	int blocksize = 256;
	int gridsize = (numObjs + blocksize*batchPerThread -1)/(blocksize*batchPerThread);
	dim3 block(blocksize, 1, 1);
	dim3 grid(gridsize, 1, 1);
	timer.start();
	distance_kernel<<<grid, block, 0, __streamId>>>(
			objs_d.getD(),
			numCoords,
			numObjs,
			targetObj_d.getD(),
			distanceObjs.getD(true),
			batchPerThread);
	cudaStreamSynchronize(__streamId);
	checkCudaErr(cudaGetLastError());
	kernelTime = timer.stop();

	timer.start();
	distanceObjs.sync();
	copyoutTime += timer.stop();


	/* find k nearest objs */
	T *distancPtr = distanceObjs.getH();
	timer.start();
	for(int i=0; i<numNN; i++){
		int min = i;
		for(int j=i+1; j<numObjs; j++){
			if(distancPtr[min]<distancPtr[j])
				min = j;
		}
		if(min != i){
			float tmp = distancPtr[i];
			distancPtr[i] = distancPtr[min];
			distancPtr[min] = tmp;
		}
		for(int j=0; j<numCoords; j++)
			objsNN[i][j] = objects[min][j];
	}
	hostCompTime = timer.stop();
	totaltime = timer0.stop();


	//runtime[0] = kernelTime + copyinTime + copyoutTime + hostCompTime;
	runtime[0] = totaltime;
	runtime[1] = kernelTime;
	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
	runtime[4] = hostCompTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

template class nnSGPU<float>;
template class nnSGPU<double>;

