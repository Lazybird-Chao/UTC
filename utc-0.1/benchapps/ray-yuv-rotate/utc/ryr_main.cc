/*
 * ryr_main.cc
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"

#include "common.h"

#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>

using namespace iUtc;

#include "task.h"
#include "cpu_task/ray_task.h"
#include "cpu_task/yuv_task.h"
#include "cpu_task/rotate_task.h"
#include "cpu_task/output_task.h"

#define MAX_THREADS 64
FTYPE aspect = 1.333333;

int main(int argc, char** grgv){
	bool printTime = false;
	char* infile_path = NULL;
	int xres=800;
	int yres=600;
	int rays_per_pixel=1;
	int loop = 100;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"w:h:i:l:vt:p:m:"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 't': nthreads=atoi(optarg);
				  break;
			case 'p': nprocess = atoi(optarg);
				  break;
			case 'm': mtype = atoi(optarg);
				  break;
			case 'i': infile_path=optarg;
					  break;
			case 'w': xres = atoi(optarg);
					  break;
			case 'h': yres = atoi(optarg);
					  break;
			case 'l': loop = atoi(optarg);
					  break;
			case ':':
				std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
				break;
			case '?':
				std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
				break;
			default:
			    break;
		}
	}
	int procs = ctx.numProcs();
	int myproc = ctx.getProcRank();
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	if(mtype==0)
		memtype = MemType::pageable;
	else if(mtype==1)
		memtype = MemType::pinned;
	else if(mtype ==2)
		memtype = MemType::unified;
	else
		std::cerr<<"wrong memory type for -m !!!"<<std::endl;

	/*
	 * read input scene file
	 */
	global_vars g_vars;
	g_vars.xres = xres;
	g_vars.yres = yres;
	g_vars.rays_per_pixel = rays_per_pixel;
	g_vars.aspect = aspect;
	sphere_array_t obj_array_for_gpu;
	vec3_t lights[MAX_LIGHTS];
	Task<SceneInit> sceneConfig(ProcList(0));
	sceneConfig.run(infile_path, &obj_array_for_gpu, &g_vars, lights);
	sceneConfig.wait();
	sphere2_t *obj_array_for_cpu = new sphere2_t[g_vars.obj_count];
	for(int i = 0; i < g_vars.obj_count; i++){
		obj_array_for_cpu[i].mat = obj_array_for_gpu.mat[i];
		obj_array_for_cpu[i].pos = obj_array_for_gpu.pos[i];
		obj_array_for_cpu[i].rad = obj_array_for_gpu.rad[i];
	}

	uint32_t *pixels_array = (uint32_t*)malloc(xres * yres * sizeof(uint32_t) * loop);

	/*
	 * raytrace task
	 */
	double ray_runtime[MAX_THREADS][3];
	int plist1[1] = {0};
	Task<crayCPUWorker> raytrace(ProcList(1, plist1), TaskType::cpu_task);

	/*
	 * yuv task
	 */
	double yuv_runtime[MAX_THREADS][3];
	int plist2[1] = {0};
	Task<YUVconvertCPUWorker> yuv(ProcList(1, plist2), TaskType::cpu_task);
	Conduit cdt1(&raytrace, &yuv);

	/*
	 * rotate task
	 */
	double rotate_runtime[MAX_THREADS][3];
	int plist3[1] = {0};
	Task<RotateCPUWorker> rotate(ProcList(1, plist3), TaskType::cpu_task);
	Conduit cdt2(&yuv, &rotate);

	//output task
	double output_runtime[MAX_THREADS][1];
	int plist4[1] = {0};
	Task<OutputWorker> output(ProcList(1, plist4), TaskType::cpu_task);
	Conduit cdt3(&rotate, &output);

	 //init tasks
	raytrace.init(g_vas, obj_array_for_cpu, pixels_array, lights, &cdt1);

	std::vector<Conduit*> cdts;
	cdts.push_back(&cdt2);
	yuv.init(xres, yres, 10, loop, &cdt1, cdts);

	rotate.init(xres, yres, &cdt2, &cdt3);

	Timer timer;
	timer.start();
	//run tasks
	int iter = 0;
	raytrace.run(ray_runtime, loop, 1);
	yuv.run(yuv_runtime);
	rotate.run(rotate_runtime, 3*loop);
	output.run(xres, yres, 3*loop, cdt3, output_runtime);

	//finish
	raytrace.wait();
	yuv.wait();
	rotate.wait();
	output.wait();

	ctx.Barrier();
	doublt totaltime = timer.stop();

	if(myProc == 0){
		std::cout<<"Test complete !!!"<<std::endl;
		/*double ray_t[3] = {ray_runtime[0][0], ray_runtime[0][1], ray_runtime[0][2]};
		double yuv_t[3] = {yuv_runtime[0][0], yuv_runtime[0][1], yuv_runtime[0][2]};
		double rotate_t[3] = {rotate_runtime[0][0], rotate_runtime[0][1], rotate_runtime[0][2]};
		double out_t[1] = {output_runtime[0][0]};
		*/
		double runtime[11] = {totaltime, ray_runtime[0][0], ray_runtime[0][1], ray_runtime[0][2],
				yuv_runtime[0][0], yuv_runtime[0][1], yuv_runtime[0][2],
				rotate_runtime[0][0], rotate_runtime[0][1], rotate_runtime[0][2],
				output_runtime[0][0]
		};
		for(int i = 0; i < 11; i++)
			runtime[i] *= 1000;
		print_time(11, runtime);

	}
	ctx.Barrier();
	return 0;



}



