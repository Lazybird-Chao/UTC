/*
 * micro_task.cc
 *
 *      Author: Chao
 */

#include "micro_task.h"
#include "micro_kernel.h"
#include "../../common/helper_err.h"
#include "cuda_runtime.h"
#include <iostream>

template<typename T>
void microTest<T>::initImpl(int nscale, int blocksize, int nStreams, int loop, enum memtype_enum memtype){
	if(__localThreadId ==0){
		std::cout<<"begin init ...\n";
		this->nscale = nscale;
		this->nStreams = nStreams;
		this->blocksize = blocksize;
		this->memtype = memtype;
		this->loop = loop;

		nsize = nscale*nscale*blocksize*nStreams*nStreams;
		streamSize = nsize/nStreams/nStreams;

		streams = new cudaStream_t[nStreams*nStreams];

	}
	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

template<typename T>
microTest<T>::~microTest(){
	delete[] streams;
}

template<typename T>
void microTest<T>::runImpl(double *runtime){
	if(__localThreadId == 0){
		std::cout<<"begin run ..."<<std::endl;
	}

	GpuKernel mykernel;
	UtcGpuContext* cur_g_ctx = mykernel.getUtcGpuContext();
	for(int i=0; i<nStreams*nStreams; i++)
		streams[i] = cur_g_ctx->getNewStream();
	/*
	 * TODO: should use some wrapper utilities to create gpu memory,
	 * transfer data and invoke kernel,
	 * as integrated with our framework.
	 *
	 * now we do these explicitly
	 */

	/*
	 * create memory
	 */
	switch(memtype){
	case pageable :
		kernel_pageable(runtime);
		break;
	case pinmem :
		kernel_pinned(runtime);
		break;
	case umem :
		kernel_umem(runtime);
		break;
	default:
		std::cout<<"wrong memtype !!!"<<std::endl;
		break;
	}

	for(int i=0; i<nStreams; i++)
		cur_g_ctx->destroyStream(streams[i]);
}

template<typename T>
void microTest<T>::kernel_pageable(double* runtime){
	Timer timer1, timer2;

	/*
	 * create memory
	 */
	data = (T*)new T[nsize];
	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaMalloc(&data_d, sizeof(T)*nsize));

	/*
	 * copy data and call kernel
	 */
	dim3 block(sqrt(blocksize), sqrt(blocksize));
	dim3 grid(sqrt(nsize)/block.x, sqrt(nsize)/block.y);
	timer1.start();
	checkCudaErr(cudaMemcpy(data_d, data, sizeof(T)*nsize,
				cudaMemcpyHostToDevice));

	timer2.start();
	micro_kernel<<<grid, block>>>(data_d, 0, nscale, loop);
	checkCudaErr(cudaGetLastError());
	checkCudaErr(cudaDeviceSynchronize());
	double kerneltime = timer2.stop();

	checkCudaErr(cudaMemcpy(data, data_d, sizeof(T)*nsize,
			cudaMemcpyDeviceToHost));
	double totaltime = timer1.stop();

	runtime[0] = totaltime;
	runtime[1] = kerneltime;


	/*
	 * scheme type I, multiple streams
	 */
	dim3 grid1(grid.x/nStreams, grid.y/nStreams);
	cudaEvent_t startEvent, stopEvent;
	checkCudaErr(cudaEventCreate(&startEvent));
	checkCudaErr(cudaEventCreate(&stopEvent));
	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaEventRecord(startEvent, 0));
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data_d[offset], &data[offset],
				streamSize*sizeof(T), cudaMemcpyHostToDevice,
									   streams[i]) );
		micro_kernel<<<grid1, block, 0, streams[i]>>>(data_d, offset, nscale, loop);
		checkCudaErr( cudaMemcpyAsync(&data[offset], &data_d[offset],
				streamSize*sizeof(T), cudaMemcpyDeviceToHost,
									   streams[i]) );
	}
	checkCudaErr(cudaEventRecord(stopEvent, 0));
	checkCudaErr(cudaEventSynchronize(stopEvent));
	float totaltime2;
	checkCudaErr(cudaEventElapsedTime(&totaltime2, startEvent, stopEvent));
	runtime[2] = (double)totaltime2/1000;

	/*
	 * scheme type II, multiple streams
	 */
	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaEventRecord(startEvent, 0));
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data_d[offset], &data[offset],
					streamSize*sizeof(T), cudaMemcpyHostToDevice,
										   streams[i]) );
	}
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		micro_kernel<<<grid1, block, 0, streams[i]>>>(data_d, offset, nscale, loop);
	}
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data[offset], &data_d[offset],
					streamSize*sizeof(T), cudaMemcpyDeviceToHost,
										   streams[i]) );
	}
	checkCudaErr(cudaEventRecord(stopEvent, 0));
	checkCudaErr(cudaEventSynchronize(stopEvent));
	float totaltime3;
	checkCudaErr(cudaEventElapsedTime(&totaltime3, startEvent, stopEvent));
	runtime[3] = (double)totaltime3/1000;

	checkCudaErr(cudaEventDestroy(startEvent));
	checkCudaErr(cudaEventDestroy(stopEvent));

	delete[] data;
	cudaFree(data_d);

}

template<typename T>
void microTest<T>::kernel_pinned(double *runtime){
	Timer timer1, timer2;
	checkCudaErr(cudaMallocHost(&data, sizeof(T)*nsize));
	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaMalloc(&data_d, sizeof(T)*nsize));

	dim3 block(sqrt(blocksize), sqrt(blocksize));
	dim3 grid(sqrt(nsize)/block.x, sqrt(nsize)/block.y);
	timer1.start();
	checkCudaErr(cudaMemcpy(data_d, data, sizeof(T)*nsize,
				cudaMemcpyHostToDevice));

	timer2.start();
	micro_kernel<<<grid, block>>>(data_d, 0, nscale, loop);
	checkCudaErr(cudaGetLastError());
	checkCudaErr(cudaDeviceSynchronize());
	double kerneltime = timer2.stop();

	checkCudaErr(cudaMemcpy(data, data_d, sizeof(T)*nsize,
				cudaMemcpyDeviceToHost));
	double totaltime = timer1.stop();

	runtime[0] = totaltime;
	runtime[1] = kerneltime;

	dim3 grid1(grid.x/nStreams, grid.y/nStreams);
	cudaEvent_t startEvent, stopEvent;
	checkCudaErr(cudaEventCreate(&startEvent));
	checkCudaErr(cudaEventCreate(&stopEvent));
	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaEventRecord(startEvent, 0));
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data_d[offset], &data[offset],
				streamSize*sizeof(T), cudaMemcpyHostToDevice,
									   streams[i]) );
		micro_kernel<<<grid1, block, 0, streams[i]>>>(data_d, offset, nscale, loop);
		checkCudaErr( cudaMemcpyAsync(&data[offset], &data_d[offset],
				streamSize*sizeof(T), cudaMemcpyDeviceToHost,
									   streams[i]) );
	}
	checkCudaErr(cudaEventRecord(stopEvent, 0));
	checkCudaErr(cudaEventSynchronize(stopEvent));
	float totaltime2;
	checkCudaErr(cudaEventElapsedTime(&totaltime2, startEvent, stopEvent));
	runtime[2] = (double)totaltime2/1000;


	memset(data, 0, sizeof(T)*nsize);
	checkCudaErr(cudaEventRecord(startEvent, 0));
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data_d[offset], &data[offset],
					streamSize*sizeof(T), cudaMemcpyHostToDevice,
										   streams[i]) );
	}
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		micro_kernel<<<grid1, block, 0, streams[i]>>>(data_d, offset, nscale, loop);
	}
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		checkCudaErr( cudaMemcpyAsync(&data[offset], &data_d[offset],
					streamSize*sizeof(T), cudaMemcpyDeviceToHost,
										   streams[i]) );
	}
	checkCudaErr(cudaEventRecord(stopEvent, 0));
	checkCudaErr(cudaEventSynchronize(stopEvent));
	float totaltime3;
	checkCudaErr(cudaEventElapsedTime(&totaltime3, startEvent, stopEvent));
	runtime[3] = (double)totaltime3/1000;

	checkCudaErr(cudaEventDestroy(startEvent));
	checkCudaErr(cudaEventDestroy(stopEvent));

	cudaFreeHost(data);
	cudaFree(data_d);
}

template<typename T>
void microTest<T>::kernel_umem(double *runtime){
	Timer timer1;
	if(getCurrentUtcGpuCtx()->getCurrentDeviceAttr(cudaDevAttrManagedMemory) ==1){
		std::cout<<"using cuda managed for umem"<<std::endl;
		checkCudaErr(cudaMallocManaged(&data_d, sizeof(T)*nsize));
	}
	else{
		std::cout<<"using pinned zero-copy for umem"<<std::endl;
		checkCudaErr(cudaMallocHost(&data_d, sizeof(T)*nsize));
	}
	memset(data_d, 0, sizeof(T)*nsize);

	dim3 block(sqrt(blocksize), sqrt(blocksize));
	dim3 grid(sqrt(nsize)/block.x, sqrt(nsize)/block.y);
	timer1.start();
	micro_kernel<<<grid, block>>>(data_d, 0, nscale, loop);
	checkCudaErr(cudaGetLastError());
	checkCudaErr(cudaDeviceSynchronize());
	double totaltime = timer1.stop();
	runtime[0] = totaltime;

	dim3 grid1(grid.x/nStreams, grid.y/nStreams);
	cudaEvent_t startEvent, stopEvent;
	checkCudaErr(cudaEventCreate(&startEvent));
	checkCudaErr(cudaEventCreate(&stopEvent));
	memset(data_d, 0, sizeof(T)*nsize);
	checkCudaErr(cudaEventRecord(startEvent, 0));
	for(int i=0; i<nStreams*nStreams; i++){
		int offset = i*streamSize;
		micro_kernel<<<grid1, block, 0, streams[i]>>>(data_d, offset, nscale, loop);
	}
	checkCudaErr(cudaEventRecord(stopEvent, 0));
	checkCudaErr(cudaEventSynchronize(stopEvent));
	float totaltime2;
	checkCudaErr(cudaEventElapsedTime(&totaltime2, startEvent, stopEvent));
	runtime[2] = (double)totaltime2/1000;

	cudaFree(data_d);
}

template class microTest<float>;
template class microTest<double>;



