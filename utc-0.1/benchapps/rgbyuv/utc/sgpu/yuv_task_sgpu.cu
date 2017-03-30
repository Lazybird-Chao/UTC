/*
 * yuv_task_sgpu.cu
 *
 *  Created on: Mar 17, 2017
 *      Author: chao
 */

#include "yuv_task_sgpu.h"
#include "yuvconvert_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>

void YUVconvertSGPU::initImpl(Image* srcImg, yuv_color_t *dstImg){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<"begin init ...\n";
		this->srcImg = srcImg;
		this->dstImg = dstImg;
		int w = srcImg->getWidth();
		int h = srcImg->getHeight();
		dstImg->y = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->u = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->v = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void YUVconvertSGPU::runImpl(double *runtime, int loop, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer;
	int w = srcImg->getWidth();
	int h = srcImg->getHeight();
	GpuData<uint8_t> img_y(w*h, memtype);
	GpuData<uint8_t> img_u(w*h, memtype);
	GpuData<uint8_t> img_v(w*h, memtype);
	GpuData<Pixel> sImg(w*h, memtype);
	sImg.initH(srcImg->getPixelBuffer());

	/*
	 * copy data in
	 */
	timer.start();
	//memcpy(sImg.getH(true), srcImg->getPixelBuffer(), sImg.getBSize());
	sImg.sync();
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
				(h+blocksize_y*batchy-1)/(blocksize_y*batchy),
				1);
	for(int i=0; i<loop; i++){
		convert<<<grid, block, 0, __streamId>>>(sImg.getD(),
				img_y.getD(true),
				img_u.getD(true),
				img_v.getD(true),
				h, w, batchx, batchy);
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

	memcpy(dstImg->y, img_y.getH(), img_y.getBSize());
	memcpy(dstImg->u, img_u.getH(), img_u.getBSize());
	memcpy(dstImg->v, img_v.getH(), img_v.getBSize());

	runtime[1] = copyinTime;
	runtime[2] = copyoutTime;
	runtime[3] = kernelTime;
	runtime[0] = copyinTime + copyoutTime + kernelTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}
