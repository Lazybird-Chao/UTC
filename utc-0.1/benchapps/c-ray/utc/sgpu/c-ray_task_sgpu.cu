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
		uint32_t *pixels,
		vec3_t *lights){
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" begin init ...\n";

		this->g_vars = g_vars;
		this->obj_array = obj_array;
		this->pixels = pixels;
		this->lights = lights;
	}

	intra_Barrier();
	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl.\n";
	}
}

void craySGPU::runImpl(double *runtime, MemType memtype){
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
	double copyinTime = timer.stop();

	/*
	 * call kernel
	 */
	size_t stacksize;
	cudaThreadGetLimit(&stacksize, cudaLimitStackSize);
	stacksize = 1024*4;
	cudaThreadSetLimit(cudaLimitStackSize, stacksize);
	dim3 block(16, 16, 1);
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
	double kernelTime = timer.stop();

	/*
	 *
	 */
	timer.start();
	pixels_d.sync();
	double copyoutTime = timer.stop();
	totaltime = timer0.stop();
	pixels_d.fetch(pixels);

	//runtime[0] = copyinTime + copyoutTime + kernelTime;
	runtime[0] = totaltime;
	runtime[1]= kernelTime;
	runtime[2]= copyinTime;
	runtime[3]= copyoutTime;

	if(__localThreadId ==0){
		std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl.\n";
	}
}

