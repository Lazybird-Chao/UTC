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
#include "gpu_task/c-ray_task_sgpu.h"

#define MAX_THREADS 64
FTYPE aspect = 1.333333;

int main(int argc, char** argv){
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
	int plist1[10] = {0,0,0,0,0,0,0,0,0,0};
	//Task<crayCPUWorker> raytrace(ProcList(10, plist1), TaskType::cpu_task);
	Task<craySGPU> raytrace(ProcList(1, plist1), TaskType::gpu_task);

	/*
	 * yuv task
	 */
	double yuv_runtime[MAX_THREADS][3];
	int plist2[4] = {0,0,0,0};
	Task<YUVconvertCPUWorker> yuv(ProcList(3, plist2), TaskType::cpu_task);
	Conduit cdt1(&raytrace, &yuv);

	/*
	 * rotate task
	 */
	double rotate_runtime[MAX_THREADS][3];
	int plist3[4] = {0,0,0,0};
	Task<RotateCPUWorker> rotate(ProcList(3, plist3), TaskType::cpu_task);
	Conduit cdt2(&yuv, &rotate);
	Task<RotateCPUWorker> rotate2(ProcList(3, plist3), TaskType::cpu_task);
	Conduit cdt22(&yuv, &rotate2);
	Task<RotateCPUWorker> rotate3(ProcList(3, plist3), TaskType::cpu_task);
	Conduit cdt23(&yuv, &rotate3);

	//output task
	double output_runtime[MAX_THREADS][3];
	int plist4[1] = {0};
	Task<OutputWorker> output(ProcList(1, plist4), TaskType::cpu_task);
	Conduit cdt3(&rotate, &output);
	Task<OutputWorker> output2(ProcList(1, plist4), TaskType::cpu_task);
	Conduit cdt32(&rotate2, &output2);
	Task<OutputWorker> output3(ProcList(1, plist4), TaskType::cpu_task);
	Conduit cdt33(&rotate3, &output3);

	 //init tasks
	//raytrace.init(g_vars, obj_array_for_cpu, pixels_array, lights, &cdt1);
	raytrace.init(g_vars, obj_array_for_gpu, pixels_array, lights, &cdt1);

	std::vector<Conduit*> cdts;
	cdts.push_back(&cdt2);
	cdts.push_back(&cdt22);
	cdts.push_back(&cdt23);
	yuv.init(xres, yres, 5, loop, &cdt1, cdts);

	rotate.init(xres, yres, &cdt2, &cdt3);
	rotate2.init(xres, yres, &cdt22, &cdt32);
	rotate3.init(xres, yres, &cdt23, &cdt33);

	Timer timer;
	timer.start();
	//run tasks
	//raytrace.run(ray_runtime, loop, 1);
	raytrace.run(ray_runtime, memtype, loop);
	yuv.run(yuv_runtime);
	rotate.run(rotate_runtime, loop);
	double tmptime[MAX_THREADS][3];
	rotate2.run(tmptime, loop);
	double tmptime2[MAX_THREADS][3];
	rotate3.run(tmptime2, loop);

	output.run(loop, &cdt3, output_runtime, 0);
	double tmptime3[MAX_THREADS][3];
	output2.run(loop, &cdt32, tmptime3, 1);
	double tmptime4[MAX_THREADS][3];
	output3.run(loop, &cdt33, tmptime4, 2);

	//finish
	raytrace.wait();
	//std::cout<<"ray finish wait"<<std::endl;
	yuv.wait();
	//std::cout<<"yuv finish wait"<<std::endl;
	rotate.wait();
	//std::cout<<"rotate finish wait"<<std::endl;
	output.wait();
	//std::cout<<"output finish wait"<<std::endl;
	output2.wait();
	output3.wait();
	ctx.Barrier();
	double totaltime = timer.stop();

	if(myproc == 0){
		std::cout<<"Test complete !!!"<<std::endl;
		std::cout<<"\tTotal time: "<<std::fixed<<std::setprecision(4)<<totaltime<<"(s)"<<std::endl;
		std::cout<<"\traytrace time: "<<std::fixed<<std::setprecision(4)<<ray_runtime[0][1]<<"(s)"<<std::endl;
		std::cout<<"\tyuv time: "<<std::fixed<<std::setprecision(4)<<yuv_runtime[0][1]<<"(s)"<<std::endl;
		std::cout<<"\trotate time: "<<std::fixed<<std::setprecision(4)<<rotate_runtime[0][1]<<"(s)"<<std::endl;
		std::cout<<"\toutput time: "<<std::fixed<<std::setprecision(4)<<output_runtime[0][1]<<"(s)"<<std::endl;
		/*double ray_t[3] = {ray_runtime[0][0], ray_runtime[0][1], ray_runtime[0][2]};
		double yuv_t[3] = {yuv_runtime[0][0], yuv_runtime[0][1], yuv_runtime[0][2]};
		double rotate_t[3] = {rotate_runtime[0][0], rotate_runtime[0][1], rotate_runtime[0][2]};
		double out_t[1] = {output_runtime[0][0]};
		*/
		double runtime[13] = {totaltime, ray_runtime[0][0], ray_runtime[0][1], ray_runtime[0][2],
				yuv_runtime[0][0], yuv_runtime[0][1], yuv_runtime[0][2],
				rotate_runtime[0][0], rotate_runtime[0][1], rotate_runtime[0][2],
				output_runtime[0][0], output_runtime[0][1], output_runtime[0][2]
		};
		for(int i = 0; i < 13; i++)
			runtime[i] *= 1000;
		print_time(13, runtime);

	}
	ctx.Barrier();
	return 0;



}



