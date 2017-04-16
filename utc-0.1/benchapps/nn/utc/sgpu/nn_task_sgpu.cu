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
	T *targetObj = new T[numCoords];
	for(int i=0; i<numCoords; i++)
		targetObj[i] = 0;


	GpuData<T> objs_d(numObjs*numCoords, memtype);
	GpuData<T> distanceObjs(numObjs, memtype);
	GpuData<T> targetObj_d(numCoords, memtype);
	objs_d.initH(objects);
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
	kernelTime1 = timer.stop();
	timer.start();

	GpuData<int> *topkIndexArray;
	int blocksize2 = 16;
	int gridsize2 = 1;
	if(numObjs > blocksize2*numNN*8)
		gridsize2 = numObjs/(blocksize2*numNN*8);
	if(gridsize2 >=16){
		topkIndexArray  = new GpuData<int>(numNN*gridsize2, memtype);
		dim3 block2(blocksize2, 1, 1);
		dim3 grid2(gridsize2, 1, 1);
		topk_kernel<<<grid2, block2, 0, __streamId>>>(
				numObjs,
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
			int min = i;
			for(int j=i+1; j<numObjs; j++){
				if(distancePtr[min]>distancePtr[j])
					min = j;
			}
			if(min != i){
				T tmp = distancePtr[i];
				distancePtr[i] = distancePtr[min];
				distancePtr[min] = tmp;
			}
			for(int j=0; j<numCoords; j++)
				objsNN[i*numCoords + j] = objects[min*numCoords + j];
		}
		hostCompTime = timer.stop();
	}
	else{
		T *distancePtr = distanceObjs.getH();
		int *topkindexPtr = topkIndexArray->getH();
		timer.start();
		for(int i=0; i<numNN; i++){
			int min = i;
			for(int j=i+1; j<topkIndexArray->getSize(); j++){
				if(distancePtr[topkindexPtr[min]]>distancePtr[topkindexPtr[j]])
					min = j;
			}
			if(min != i){
				int tmp = topkindexPtr[min];
				topkindexPtr[min] = topkindexPtr[i];
				topkindexPtr[i] = tmp;
			}
			for(int j=0; j<numCoords; j++){
				objsNN[i*numCoords +j] = objects[topkindexPtr[i]*numCoords+j];
			}
		}
		hostCompTime = timer.stop();
	}

	totaltime = timer0.stop();


	//runtime[0] = kernelTime + copyinTime + copyoutTime + hostCompTime;
	runtime[0] = totaltime;
	runtime[1] = kernelTime1;
	runtime[2] = kernelTime2;
	runtime[3] = copyinTime;
	runtime[4] = copyoutTime;
	runtime[5] = hostCompTime;
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}

template class nnSGPU<float>;
template class nnSGPU<double>;

