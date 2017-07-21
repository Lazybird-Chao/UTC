/*
 * yuv_task_sgpu.cu
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#include "yuv_task_mgpu.h"
#include "yuvconvert_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>

void YUVconvertMGPU::initImpl(Image* srcImg, yuv_color_t *dstImg){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";
		this->srcImg = srcImg;
		this->dstImg = dstImg;
		int w = srcImg->getWidth();
		int h = srcImg->getHeight();
		dstImg->y = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->u = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->v = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	}
	intra_Barrier();
	int w = srcImg->getWidth();
	int h = srcImg->getHeight();
	int rows_per_thread = h/__numLocalThreads;
	if(__localThreadId < h % __numLocalThreads){
		num_rows = rows_per_thread+1;
		start_row = __localThreadId*(rows_per_thread+1);
		end_row = start_row + (rows_per_thread+1) -1;
	}
	else{
		num_rows = rows_per_thread;
		start_row = __localThreadId*rows_per_thread + h % __numLocalThreads;
		end_row = start_row + rows_per_thread -1;
	}
	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void YUVconvertMGPU::runImpl(double runtime[][4], int loop, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer, timer0;
	double totaltime;

	int w = srcImg->getWidth();
	int h = srcImg->getHeight();
	GpuData<uint8_t> img_y(w*num_rows, memtype);
	GpuData<uint8_t> img_u(w*num_rows, memtype);
	GpuData<uint8_t> img_v(w*num_rows, memtype);
	GpuData<Pixel> partial_sImg(w*num_rows, memtype);
	Pixel *img_startPtr = srcImg->getPixelBuffer() + start_row*w;
	partial_sImg.initH(img_startPtr);

	/*
	 * copy data in
	 */
	timer0.start();
	timer.start();
	//memcpy(sImg.getH(true), srcImg->getPixelBuffer(), sImg.getBSize());
	partial_sImg.sync();
	double copyinTime = timer.stop();

	/*
	 * invoke kernel
	 */
	timer.start();
	int blocksize_x = 32;
	int blocksize_y = 16;
	int batchx = 1;
	int batchy = 1;
	dim3 block(blocksize_x, blocksize_y, 1);
	dim3 grid((w+blocksize_x*batchx-1)/(blocksize_x*batchx),
				(num_rows+blocksize_y*batchy-1)/(blocksize_y*batchy),
				1);
	for(int i=0; i<loop; i++){
		convert<<<grid, block, 0, __streamId>>>(partial_sImg.getD(),
				img_y.getD(true),
				img_u.getD(true),
				img_v.getD(true),
				num_rows.load(), w, batchx, batchy);
		checkCudaErr(cudaGetLastError());
		//checkCudaErr(cudaDeviceSynchronize());
		checkCudaErr(cudaStreamSynchronize(__streamId));
	}
	double kernelTime = timer.stop();


	/*
	 * copy data out
	 */
	timer.start();
	img_y.sync();
	img_u.sync();
	img_v.sync();
	//std::cout<<img_y.at(10)<<std::endl;
	double copyoutTime = timer.stop();


	memcpy(dstImg->y + w*start_row, img_y.getH(), img_y.getBSize());
	memcpy(dstImg->u + w*start_row, img_u.getH(), img_u.getBSize());
	memcpy(dstImg->v + w*start_row, img_v.getH(), img_v.getBSize());
	totaltime = timer0.stop();

	runtime[__localThreadId][2] = copyinTime;
	runtime[__localThreadId][3] = copyoutTime;
	runtime[__localThreadId][1] = kernelTime;
	//runtime[0] = copyinTime + copyoutTime + kernelTime;
	runtime[__localThreadId][0] = totaltime;

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}
