/*
 * c-ray_task_sgpu.cu
 *
 *  Created on: Mar 24, 2017
 *      Author: chao
 */
#include "c-ray_task_sgpu.h"
#include "c-ray_kernel_v2.h"
#include "../../../common/helper_err.h"

__device__  global_vars g_vars_d;
__device__  vec3_t lights_d[MAX_LIGHTS];
__device__  vec2_t urand_d[NRAN];
__device__  int irand_d[NRAN];

void craySGPU::initImpl(global_vars g_vars,
		sphere_array_t obj_array,
		uint32_t *pixels_array,
		vec3_t *lights,
		Conduit *cdtOut){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";

		this->g_vars = g_vars;
		this->obj_array = obj_array;
		this->pixels_array = pixels_array;
		this->lights = lights;
		this->m_cdtOut = cdtOut;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void craySGPU::runImpl(double runtime[][3], MemType memtype, int loop){
	if(__localThreadId == 0){
		std::cout<<getCurrentTask()->getName()<<" begin run ..."<<std::endl;
	}
	Timer timer, timer0;
	double totaltime;

	int xres = g_vars.xres;
	int yres = g_vars.yres;
	GpuData<unsigned int> pixels_d(xres*yres);
	GpuData<vec3_t> obj_array_pos(g_vars.obj_count);
	GpuData<material_t> obj_array_mat(g_vars.obj_count);
	GpuData<FTYPE> obj_array_rad(g_vars.obj_count);
	obj_array_pos.initH(obj_array.pos);
	obj_array_mat.initH(obj_array.mat);
	obj_array_rad.initH(obj_array.rad);

	vec2_t urand[NRAN];
	int irand[NRAN];
	for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

	/*
	 * data in
	 */
	timer0.start();
	timer.start();
	obj_array_pos.sync();
	obj_array_mat.sync();
	obj_array_rad.sync();

	//std::cout<<xres<<" "<<yres<<std::endl;

	checkCudaErr(
			cudaMemcpyToSymbol(lights_d, lights, sizeof(vec3_t)*MAX_LIGHTS, 0,
					cudaMemcpyHostToDevice));
	checkCudaErr(
			cudaMemcpyToSymbol(urand_d, urand, sizeof(vec2_t)*NRAN, 0,
					cudaMemcpyHostToDevice));
	checkCudaErr(
			cudaMemcpyToSymbol(irand_d, irand, sizeof(int)*NRAN, 0,
					cudaMemcpyHostToDevice));
	checkCudaErr(
			cudaMemcpyToSymbol(g_vars_d, (void*)&g_vars, sizeof(g_vars), 0,
					cudaMemcpyHostToDevice));
	double commtime = timer.stop();
	double comptime = 0;

	int iter = 0;
	while(iter < loop){
		pixels_d.setH();
		size_t stacksize;
		cudaThreadGetLimit(&stacksize, cudaLimitStackSize);
		//std::cout<<stacksize<<std::endl;
		stacksize = 1024*4;
		cudaThreadSetLimit(cudaLimitStackSize, stacksize);
		dim3 block(16, 8, 1);
		dim3 grid((xres+block.x-1)/block.x, (yres+block.y-1)/block.y,1);
		timer.start();
		render_kernel<<<grid, block, 0, __streamId>>>(
				pixels_d.getD(true),
				obj_array_pos.getD(),
				obj_array_mat.getD(),
				obj_array_rad.getD()
				);
		checkCudaErr(cudaGetLastError());
		checkCudaErr(cudaStreamSynchronize(__streamId));
		comptime += timer.stop();

		timer.start();
		pixels_d.sync();
		pixels_d.fetch(pixels_array + iter*xres*yres);
		/*
		for(int i =0; i<yres; i++)
			for(int j=0; j<xres; j++)
				if(i==10 && j%100==0)
					std::cout<<i<<" "<<j<<"  "<<pixels_array[iter*xres*yres+i*xres+j]<<std::endl;;
					*/
		m_cdtOut->WriteBy(0, pixels_array+iter*xres*yres,
				sizeof(uint32_t)*xres*yres, iter);
		__fastIntraSync.wait();
		commtime += timer.stop();
		iter++;
	}

	totaltime = timer0.stop();
	//runtime[0] = copyinTime + copyoutTime + kernelTime;
	runtime[__localThreadId][0] = totaltime;
	runtime[__localThreadId][1]= comptime;
	runtime[__localThreadId][2]= commtime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

