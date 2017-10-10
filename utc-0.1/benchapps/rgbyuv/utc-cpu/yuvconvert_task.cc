/*
 * yuvconversion_task.cc
 *
 *  Created on: Oct 9, 2017
 *      Author: Chao
 */

#include "task.h"
#include <iostream>
#include <cmath>

void YUVconvertWorker::initImpl(Image* srcImg, yuv_color_t *dstImg){
	if(__localThreadId == 0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";
		this->srcImg = srcImg;
		this->dstImg = dstImg;
		int w = srcImg->getWidth();
		int h = srcImg->getHeight();
		dstImg->y = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->u = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
		dstImg->v = (uint8_t*)malloc(sizeof(uint8_t)*w*h);
	}
	__fastIntraSync.wait();
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
	__fastIntraSync.wait();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

void YUVconvertWorker::runImpl(double runtime[][1], int loop){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer;
	int w = srcImg->getWidth();
	int h = srcImg->getHeight();

	timer.start();
	uint8_t R,G,B,Y,U,V;
	for(int i=0; i<loop; i++){
		uint8_t *pY = dstImg->y;
		uint8_t *pU = dstImg->u;
		uint8_t *pV = dstImg->v;
		Pixel *in = srcImg->getPixelBuffer();
		for(int j=start_row; j <= end_row; j++){
			for(int k = 0; k < w; k++){
				R = in[j*w + k].r;
				G = in[j*w + k].g;
				B = in[j*w + k].b;
				Y = (uint8_t)round(0.256788*R+0.504129*G+0.097906*B) + 16;
				U = (uint8_t)round(-0.148223*R-0.290993*G+0.439216*B) + 128;
				V = (uint8_t)round(0.439216*R-0.367788*G-0.071427*B) + 128;
				pY[j*w + k] = Y;
				pU[j*w + k] = U;
				pV[j*w + k] = V;
			}
		}
	}
	double total = timer.stop();
	runtime[__localThreadId][0] = total;

	inter_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}


