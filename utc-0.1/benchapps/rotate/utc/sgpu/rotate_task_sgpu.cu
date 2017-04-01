/*
 * rotate_task_sgpu.cu
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 */
#include "rotate_task_sgpu.h"
#include "rotate_kernel.h"
#include "../../../common/helper_err.h"
#include <iostream>



void RotateSGPU::initImpl(Image* srcImg, Image* dstImg, int angle){
	if(__localThreadId ==0){
		std::cout<<"begin init ...\n";
		this->srcImg = srcImg;
		this->dstImg = dstImg;
		this->angle = angle;

		/*
		 * compute the out image's size
		 */
		float2 ul, ur, ll, lr;
		float xc = (float)srcImg->getWidth() / 2.0;
		float yc = (float)srcImg->getHeight() / 2.0;
		ul.x = -xc;
		ul.y = yc;
		ur.x = xc;
		ur.y = yc;
		ll.x = -xc;
		ll.y = -yc;
		lr.x = xc;
		lr.y = -yc;
		float2 outCorner[4];
		rotatePoint(ul, outCorner[0], angle);
		rotatePoint(ur, outCorner[1], angle);
		rotatePoint(ll, outCorner[2], angle);
		rotatePoint(lr, outCorner[3], angle);
		//compute the out image's size
		float maxW = outCorner[0].x;
		float minW = outCorner[0].x;
		float maxH = outCorner[0].y;
		float minH = outCorner[0].y;
		for(int i=1; i<4; i++){
			if(outCorner[i].x > maxW)
				maxW = outCorner[i].x;
			if(outCorner[i].x< minW)
				minW = outCorner[i].x;
			if(outCorner[i].y > maxH)
				maxH = outCorner[i].y;
			if(outCorner[i].y< minH)
				minH = outCorner[i].y;
		}
		int outH = (int)maxH-minH;
		int outW = (int)maxW-minW;
		dstImg->createImageFromTemplate(outW, outH, srcImg->getDepth());

	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void RotateSGPU::runImpl(double *runtime, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer, timer0;
	double totaltime;

	GpuData<Pixel> sImg(srcImg->getWidth()*srcImg->getHeight(), memtype);
	GpuData<Pixel> dImg(dstImg->getWidth()*dstImg->getHeight(), memtype);
	sImg.initH(srcImg->getPixelBuffer());
	//std::cout<<srcImg->getWidth()<<" "<<srcImg->getHeight()<<" "<<sizeof(Pixel)<<" "<<sImg.getBSize()<<std::endl;

	//std::cout<<srcImg->getWidth()<<" "<<srcImg->getHeight()<<sImg.getBSize()<<std::endl;

	/*
	 * copy data in
	 */
	timer0.start();
	timer.start();
	//memcpy(sImg.getH(true), srcImg->getPixelBuffer(), sImg.getBSize());
	sImg.syncH();
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
	dim3 grid((dstImg->getWidth()+blocksize_x*batchx-1)/(blocksize_x*batchx),
				(dstImg->getHeight()+blocksize_y*batchy-1)/(blocksize_y*batchy),
				1);
	rotate_kernel<<<grid, block, 0, __streamId>>>(sImg.getD(),
									srcImg->getWidth(),
									srcImg->getHeight(),
									dImg.getD(true),
									dstImg->getWidth(),
									dstImg->getHeight(),
									angle,
									batchx,
									batchy);
	checkCudaErr(cudaGetLastError());
	//checkCudaErr(cudaDeviceSynchronize());
	checkCudaErr(cudaStreamSynchronize(__streamId));
	double kernelTime = timer.stop();

	/*
	 * copy data out
	 */
	timer.start();
	dImg.syncD();
	//Pixel tmp = dImg.at(1000);
	//std::cout<<tmp.r<<std::endl;
	double copyoutTime = timer.stop();
	totaltime = timer0.stop();
	memcpy(dstImg->getPixelBuffer(), dImg.getH(), dImg.getBSize());


	runtime[2] = copyinTime;
	runtime[3] = copyoutTime;
	runtime[1] = kernelTime;
	//runtime[0] = copyinTime+copyoutTime+kernelTime;
	runtime[0] = totaltime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}


