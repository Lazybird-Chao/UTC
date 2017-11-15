/*
 * yuv_task.cc
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#include "yuv_task.h"
#include "../common.h"
#include <iostream>
#include <cmath>

void YUVconvertCPUWorker::initImpl(int w, int h, int innerloop, int loop,
		iUtc::Conduit *cdtIn, std::vector<iUtc::Conduit*> cdtOut){
	if(__localThreadId == 0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";
		this->w = w;
		this->h = h;
		this->innerloop = innerloop;
		this->loop = loop;
		this->cdtIn = cdtIn;
		this->cdtOut = cdtOut;

		srcImg_array = new uint32_t[w*h*loop];
		y_array = new uint8_t[w*h*loop];
		u_array = new uint8_t[w*h*loop];
		v_array = new uint8_t[w*h*loop];
	}
	__fastIntraSync.wait();
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

void YUVconvertCPUWorker::runImpl(double runtime[][3]){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}

	Timer timer, timer0;
	double comptime = 0;
	double commtime = 0;
	uint8_t R,G,B,Y,U,V;

	timer0.start();
	int iter = 0;
	while(iter < loop){
		uint32_t *pixels = srcImg_array + w*h*iter;
		timer.start();
		cdtIn->ReadBy(0, pixels, w*h*sizeof(uint32_t), iter);
		__fastIntraSync.wait();
		commtime += timer.stop();

		timer.start();
		for(int i=0; i<innerloop; i++){
			uint8_t *pY = y_array + w*h*iter;
			uint8_t *pU = u_array + w*h*iter;
			uint8_t *pV = v_array + w*h*iter;
			for(int j=start_row; j <= end_row; j++){
				for(int k = 0; k < w; k++){
					R = (pixels[j*w+k]>>RSHIFT) & 0xff;
					G = (pixels[j*w+k]>>GSHIFT) & 0xff;
					B = (pixels[j*w+k]>>BSHIFT) & 0xff;
					Y = (uint8_t)round(0.256788*R+0.504129*G+0.097906*B) + 16;
					U = (uint8_t)round(-0.148223*R-0.290993*G+0.439216*B) + 128;
					V = (uint8_t)round(0.439216*R-0.367788*G-0.071427*B) + 128;
					pY[j*w + k] = Y;
					pU[j*w + k] = U;
					pV[j*w + k] = V;
				}
			}
		}
		__fastIntraSync.wait();
		comptime += timer.stop();

		timer.start();
		for(int i = 0; i<cdtOut.size(); i++){
			cdtOut[i]->WriteBy(0, pY, w*h*sizeof(uint8_t), iter*cdtOut.size()+i);
		}
		__fastIntraSync.wait();
		commtime += timer.stop();
	}
	double total = timer0.stop();

	runtime[__localThreadId][0] = total;
	runtime[__localThreadId][1] = commtime;
	runtime[__localThreadId][2] = comptime;

	inter_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}


