/*
 * ray_task.cc
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#include "ray_task.h"
#include <cmath>

thread_local int crayWorker::local_yres;
thread_local int crayWorker::local_startYresIndex;

void crayCPUWorker::initImpl(global_vars g_vars,
		sphere2_t* obj_array,
		uint32_t *pixels_array,
		vec3_t *lights,
		iUtc::Conduit *cdtOut){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";

		this->g_vars = g_vars;
		this->obj_array = obj_array;
		this->pixels_array = pixels_array;
		this->lights = lights;
		this->m_cdtOut = cdtOut;
	}
	__fastIntraSync.wait();
	int yresPerThread = g_vars.yres / __numLocalThreads;
	if(__localThreadId < g_vars.yres % __numLocalThreads){
		local_yres = yresPerThread +1;
		local_startYresIndex = __localThreadId *(yresPerThread+1);
	}
	else{
		local_yres = yresPerThread;
		local_startYresIndex = __localThreadId*yresPerThread + g_vars.yres % __numLocalThreads;
	}
	__fastIntraSync.wait();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}

}

void crayCPUWorker::runImpl(double runtime[][3], int loop, bool needDoOutput){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	int xres = g_vars.xres;	//column
	int yres = g_vars.yres;	//row

	for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

	__fastIntraSync.wait();

	int iter = 0;
	double comptime = 0;
	double commtime = 0;
	timer0.start();
	while(iter < loop){
		timer.start();
		uint32_t *pixels = pixels_array + iter*xres*yres;
		for(int i = local_startYresIndex; i < local_startYresIndex+local_yres; i++){
			render_scanline(xres,
							yres,
							i,
							(uint32_t*)(pixels+i*xres),
							g_vars.rays_per_pixel);
		}
		comptime += timer.stop();

		timer.start();
		if(needDoOutput && m_cdtOut != null){
			m_cdtOut->WriteBy(0, pixels, sizeof(uint32_t)*xres*yres, iter);
		}
		__fastIntraSync.wait();
		commtime += timer.stop();
		iter++;
	}
	double totaltime = timer0.stop();

	__fastIntraSync.wait();
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1] = comptime;
	runtime[__localThreadId][2] = commtime;
}


