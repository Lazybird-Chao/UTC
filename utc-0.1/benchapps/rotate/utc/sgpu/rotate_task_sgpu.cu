/*
 * rotate_task_sgpu.cu
 *
 *  Created on: Mar 15, 2017
 *      Author: chao
 */
#include "rotate_task_sgpu.h"
#include <iostream>
#include <fstream>

void ImageCreate::runImpl(Image* srcImg, char* infile_path){
	srcImg->createImageFromFile(infile_path);
}

void ImageOut::runImpl(Image * img, char* outfile_path){
	std::fstream out;
	out.open(outfile_path, std::fstream::out);
	if(!out.is_open()){
		std::cerr<<"Error, cannot create output file!"<<std::endl;
		return 1;
	}
	if(img->getDepth() == 3){
		out<<"P6\n";
	}
	else{
		std::cerr<<"Error, unsupported image file format!"<<std::endl;
		return 1;
	}
	out << img->getWidth() << " " << img->getHeight() << "\n" << img->getMaxcolor() << "\n";
	for(int i = 0; i < img->getHeight(); i++) {
		for(int j = 0; j < img->getWidth(); j++) {
			Pixel p = img->getPixelAt(j, i);
			out.put(p.r);
			out.put(p.g);
			out.put(p.b);
		}
	}
	out.close();

}

void Rotate::initImpl(Image* srcImg, Image* dstImg){
	if(__localThreadId ==0){
		std::cout<<"begin init ...\n";
		this->srcImg = srcImg;
		this->dstImg = dstImg;

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

void Rotate::runImpl(int angle, double *runtime, MemType memtype){
	if(__localThreadId == 0){
		std::cout<<"begin run ..."<<std::endl;
	}

	Timer timer;
	GpuData<Pixel> sImg(srcImg->getWidth()*srcImg->getHeight(), memtype);
	GpuData<Pixel> dImg(dstImg->getWidth()*dstImg->getHeight(), memtype);
	memcpy(sImg.getH(true), srcImg->getPixelBuffer(), sImg.getBSize());

	/*
	 * copy data in
	 */
	timer.start();
	sImg.syncH();
	double copyinTime = timer.stop();

	/*
	 * invoke kernel
	 */
	timer.start();
	int blocksize = 16;
	int batchx = 1;
	int batchy = 1;
	dim3 block(blocksize, blocksize, 1);
	dim3 grid((outW+blocksize*batchx-1)/(blocksize*batchx),
				(outH+blocksize*batchy-1)/(blocksize*batchy),
				1);
	rotate_kernel<<<grid, block>>>(sImg.getD(),
									srcImg->getWidth(),
									srcImg->getHeight(),
									dImg.getD(true),
									dstImg->getWidth(),
									dstImg->getHeight(),
									angle,
									batchx,
									batchy);
	checkCudaErr(cudaGetLastError());
	checkCudaErr(cudaDeviceSynchronize());
	double kernelTime = timer.stop();

	/*
	 * copy data out
	 */
	timer.start();
	dImg.syncD();
	double copyoutTime = timer.stop();

	memcpy(dstImg->getPixelBuffer(), dImg.getH(), dImg.getBSize());

	runtime[1] = copyinTime;
	runtime[2] = copyoutTime;
	runtime[3] = kernelTime;
	runtime[0] = copyinTime+copyoutTime+kernelTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}

}


