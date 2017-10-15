/*
 * nbody_main.cc
 *
 *  Created on: Oct 14, 2017
 *      Author: Chao
 */
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
#include "./gpu/nbody_task.h"

#define FTYPE double
#define MAX_THREADS 64
#define MAX_TIMER 9

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
	int nthreads=1;
	int nprocess=1;

	int blocksize = 256;


	/* initialize UTC context */
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vt:p:i:l:o:O:n:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 't': nthreads=atoi(optarg);
			  break;
		case 'p': nprocess = atoi(optarg);
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
		std::cerr<<"only run one thread on a node for task !!!\n";
		return 1;
	}
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

	/*
	 * create random initial body data sets
	 */
	FTYPE *body_pos = nullptr;
	FTYPE *body_vel = nullptr;
	float *body_color = nullptr;
	if(myproc == 0){
		body_pos = new FTYPE[numBodies*4];
		body_vel = new FTYPE[numBodies*4];
		body_color = new float[numBodies*4]; //not used here actually, used for disply
	}
	Task<NbodyInit<FTYPE>> bodyInit(ProcList(0));
	bodyInit.run(NBODY_CONFIG_SHELL, body_pos, body_vel, body_color,
			activeParams.m_clusterScale, activeParams.m_velocityScale,
			numBodies, true);
	bodyInit.wait();
	FTYPE *body_outBuffer = nullptr;
	if(myproc == 0){
		body_outBuffer = new FTYPE[numBodies*4*(iteration/outInterval + 1)];
	// copy the initial pos info to out
		memcpy(body_outBuffer, body_pos, numBodies*4*sizeof(FTYPE));
	}

	/*
	 * run Nbody process
	 */
	ProcList plist;
	for(int i = 0; i<procs; i++)
		for(int j = 0; j<nthreads; j++)
			plist.push_back(i);
	Task<BodySystem<FTYPE>> nbody(plist, TaskType::cpu_task);
	nbody.init(numBodies,
			(FTYPE)activeParams.m_softening*(FTYPE)activeParams.m_softening,
			activeParams.m_damping,
			body_pos,
			body_vel);
	double runtime_m[MAX_THREADS][MAX_TIMER];
	nbody.run(runtime_m,
			iteration,
			outInterval,
			blocksize,
			activeParams.m_timestep,
			body_outBuffer+numBodies*4
			);
	nbody.wait();
	/*
	 * output to a file
	 */
	if(outfilename){
		FILE *fp = fopen(outfilename, "w");
		if(!fp){
			std::cout<<"Cann't open the output file !!!"<<std::endl;
			exit(1);
		}
		Task<Output<FTYPE>> outtask(ProcList(0));
		for(int i=0; i<iteration/outInterval+1; i++){
			FTYPE timestamp = i*activeParams.m_timestep;
			outtask.run(&fp, &body_outBuffer[i*numBodies*4], timestamp, numBodies);
			outtask.wait();
		}
		fclose(fp);
	}
	if(myproc == 0){
		delete[] body_pos;
		delete[] body_vel;
		delete[] body_color;
		delete[] body_outBuffer;
	}

	if(myproc == 0){
		double runtime[MAX_TIMER]={0,0,0,0,0,0,0,0,0};
		for(int i=0; i<nthreads; i++)
			for(int j=0; j<MAX_TIMER; j++)
				runtime[j]+= runtime_m[i][j];
		for(int j=0; j<MAX_TIMER; j++)
			runtime[j] /= nthreads;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tbody system info:"<<std::endl;
			std::cout<<"\t\tTotal bodies : "<<numBodies<<std::endl;
			std::cout<<"\t\tIterations: "<<iteration<<std::endl;
			std::cout<<"\t\tOutput frames: "<<iteration/outInterval + 1<<std::endl;
			//std::cout<<"\t\tThreads per body: "<<threadsperBody<<std::endl;
			std::cout<<"\ttime info: "<<std::endl;
			std::cout<<"\t\ttotal runtime: "<<std::fixed<<std::setprecision(4)
					<<1000*(runtime[0])
					<<"(ms)"<<std::endl;
			std::cout<<"\t\tcompute time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
			std::cout<<"\t\tcommtime time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
			std::cout<<"\t\tcopytime time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[3]<<"(ms)"<<std::endl;
		}

		for(int i=0; i<4; i++)
			runtime[i] *= 1000;
		print_time(4, runtime);

	}
	ctx.Barrier();
	return 0;

}

























