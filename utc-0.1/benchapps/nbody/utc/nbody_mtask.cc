/*
 * nbody_mtask.cc
 *
 *  Created on: Apr 4, 2017
 *      Author: chao
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_err.h"
#include "../../common/helper_printtime.h"
#include "Utc.h"
#include "UtcGpu.h"
using namespace iUtc;

#include "task.h"
#include "sgpu/body_task_sgpu.h"

#define FTYPE double

struct NBodyParams
{
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;

    void print()
    {
        printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n",
               m_timestep, m_clusterScale, m_velocityScale,
               m_softening, m_damping, m_pointSize, m_x, m_y, m_z);
    }
};

NBodyParams demoParams[] =
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    { 0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0, -50},
};

int main(int argc, char** argv){
	bool printTime = false;
	int paraSelect  = 0;
	int iteration = 1;
	int outInterval = 0;
	int numBodies = 1024;
	char* outfilename = NULL;

	int blocksize = 256;
	int ntasks = 1;

	int nthreads=1;
	int nprocess=1;

	MemType memtype = MemType::pageable;
	int mtype = 0;

	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vt:p:m:i:l:o:O:n:b:T:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
			  break;
		case 'm': mtype = atoi(optarg);
			  break;
		case 'i':
			paraSelect = atoi(optarg);
			break;
		case 'o':
			outInterval = atoi(optarg);
			break;
		case 'O':
			outfilename = optarg;
			break;
		case 'l':
			iteration = atoi(optarg);
			break;
		case 'n':
			numBodies = atoi(optarg);
			break;
		case 'b':
			blocksize = atoi(optarg);
			break;
		case 'T':
			ntasks = atoi(optarg);
			break;
		case '?':
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
	if(nthreads != 1){
		std::cerr<<"require one thread !!!\n";
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

	if(outInterval==0)
		outInterval = iteration;
	if(paraSelect > 6)
		paraSelect = 0;
	NBodyParams activeParams = demoParams[paraSelect];

	if(numBodies%blocksize!=0){
		std::cout<<"number of bodies is scaled to multiple of "<<blocksize<<std::endl;
		numBodies = (numBodies/blocksize +1)*blocksize;
	}
	if (numBodies <= 1024)
	{
		activeParams.m_clusterScale = 1.52f;
		activeParams.m_velocityScale = 2.f;
	}
	else if (numBodies <= 2048)
	{
		activeParams.m_clusterScale = 1.56f;
		activeParams.m_velocityScale = 2.64f;
	}
	else if (numBodies <= 4096)
	{
		activeParams.m_clusterScale = 1.68f;
		activeParams.m_velocityScale = 2.98f;
	}
	else if (numBodies <= 8192)
	{
		activeParams.m_clusterScale = 1.98f;
		activeParams.m_velocityScale = 2.9f;
	}
	else if (numBodies <= 16384)
	{
		activeParams.m_clusterScale = 1.54f;
		activeParams.m_velocityScale = 8.f;
	}
	else if (numBodies <= 32768)
	{
		activeParams.m_clusterScale = 1.44f;
		activeParams.m_velocityScale = 11.f;
	}


	std::vector<FTYPE*> body_pos_vec;
	std::vector<FTYPE*> body_vel_vec;
	std::vector<FTYPE*> body_outBuffer_vec;
	for(int i=0; i<ntasks; i++){
		body_pos_vec.push_back(new FTYPE[numBodies*4]);
		body_vel_vec.push_back(new FTYPE[numBodies*4]);
		body_outBuffer_vec.push_back(new FTYPE[numBodies*4*(iteration/outInterval + 1)]);
	}
	Task<NbodyInit<FTYPE>> bodyInit(ProcList(0));
	for(int i=0; i<ntasks; i++){
		bodyInit.run(NBODY_CONFIG_SHELL, body_pos_vec.at(i), body_vel_vec.at(i), nullptr,
				activeParams.m_clusterScale, activeParams.m_velocityScale,
				numBodies, true);
		bodyInit.wait();
		memcpy(body_outBuffer_vec.at(i), body_pos_vec.at(i), numBodies*4*sizeof(FTYPE));
	}

	std::vector<Task<BodySystemSGPU<FTYPE>>*> nbody_vec;
	for(int i=0; i<ntasks; i++){
		Task<BodySystemSGPU<FTYPE>> *nbody = new Task<BodySystemSGPU<FTYPE>>(ProcList(0), TaskType::gpu_task);
		nbody->init(numBodies,
				(FTYPE)activeParams.m_softening*(FTYPE)activeParams.m_softening,
				activeParams.m_damping,
				body_pos_vec.at(i),
				body_vel_vec.at(i));
		nbody_vec.push_back(nbody);
	}
	for(int i=ntasks; i<48; i++){
		Task<BodySystemSGPU<FTYPE>> *nbody = new Task<BodySystemSGPU<FTYPE>>(ProcList(0));
		nbody_vec.push_back(nbody);
	}

	Timer timer;
	timer.start();
	std::vector<double*> runtime_vec;
	for(int i=0; i<ntasks; i++){
		runtime_vec.push_back(new double[4]);
		Task<BodySystemSGPU<FTYPE>> *nbody = nbody_vec.at(i);
		nbody->run(runtime_vec.at(i),
				iteration,
				outInterval,
				blocksize,
				activeParams.m_timestep,
				body_outBuffer_vec.at(i)+numBodies*4,
				memtype);
	}
	for(int i=0; i<ntasks; i++){
		Task<BodySystemSGPU<FTYPE>> *nbody = nbody_vec.at(i);
		nbody->wait();
	}
	double totaltime = timer.stop();
	//std::cout<<totaltime<<std::endl;

	for(int i=0; i<ntasks; i++){
		delete nbody_vec.at(i);
		delete body_pos_vec.at(i);
		delete body_vel_vec.at(i);
		//std::cout<<i<<std::endl;
	}
	for(int i=ntasks; i<48; i++)
		delete nbody_vec.at(i);

	double runtime[5] ={0,0,0,0,0};
	for(int i=0; i<ntasks; i++){
		double *tmp = runtime_vec.at(i);
		for(int j=0; j<4; j++)
			runtime[j] += tmp[j];
	}

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tbody system info:"<<std::endl;
		std::cout<<"\t\tTotal bodies : "<<numBodies<<std::endl;
		std::cout<<"\t\tIterations: "<<iteration<<std::endl;
		std::cout<<"\t\tOutput frames: "<<iteration/outInterval + 1<<std::endl;
		std::cout<<"\t\tNumber of tasks: "<<ntasks<<std::endl;
		std::cout<<"\ttime info: "<<std::endl;
		std::cout<<"\t\treal total runtime: "<<std::fixed<<std::setprecision(4)<<1000*totaltime<<std::endl;
		std::cout<<"\t\ttotal runtime: "<<std::fixed<<std::setprecision(4)
				<<1000*(runtime[0])
				<<"(ms)"<<std::endl;
		std::cout<<"\t\tkernel time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopyin time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		std::cout<<"\t\tcopyout time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;
	}

	runtime[4] = totaltime*1000;
	for(int i=0; i<4; i++)
		runtime[i] = runtime[i]/ntasks*1000;
	print_time(5, runtime);

	return 0;

}








