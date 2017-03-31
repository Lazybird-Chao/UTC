/*
 * nbody_main.cc
 *
 *      Author: Chao
 *
 * The sequential version of N-body system simulation. Refer cuda sdk's nbody demo.
 * Randomly generate N bodies' initial position and do iteration. Output body's new
 * position info(a frame) for every several iterations.
 *
 * usage:
 * 		compile with Makefile.
 * 		run as: ./a.out -v -i 0 -l 100 -o 10 -n 1024
 * 			-v: print time info
 * 			-i: select which input parameter for simulation, 7 preset parameter sets for choosing.
 * 			-l: the number of iterations to run
 * 			-o: the interval of iterations to gather new position data
 * 			-n: total number of bodies for simulation
 *
 *
 *
 */


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "nbody.h"
#include "bodysystem.h"

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


int main(int argc, char* argv[]){
	bool printTime = false;
	int paraSelect  = 0;
	int iteration = 1;
	int outInterval = 0;
	int numBodies = 1024;
	char* outfilename= NULL;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vi:l:o:O:n:"))!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
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
	if(outInterval ==0)
		outInterval = iteration;

	if(paraSelect > 6)
		paraSelect = 0;
	NBodyParams activeParams = demoParams[paraSelect];
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
	FTYPE *body_pos = new FTYPE[numBodies*4];
	FTYPE *body_vel = new FTYPE[numBodies*4];
	float *body_color = new float[numBodies*4]; //not used here actually, used for disply
	randomizeBodies(NBODY_CONFIG_SHELL, body_pos, body_vel, body_color,
			activeParams.m_clusterScale, activeParams.m_velocityScale,
			numBodies, true);
	FTYPE *body_outBuffer = new FTYPE[numBodies*4*(iteration/outInterval + 1)];
	// copy the initial pos info to out
	memcpy(body_outBuffer, body_pos, numBodies*4*sizeof(FTYPE));


	/*
	 * create bodysystem
	 */
	BodySystem<FTYPE> nbody(numBodies);
	nbody.setDamping(activeParams.m_damping);
	nbody.setSoftening(activeParams.m_softening);
	nbody.setPosArray(body_pos);
	nbody.setVelArray(body_vel);

	/*
	 * iterate body system
	 */
	std::cout<<"start iteration ..."<<std::endl;
	double t1, t2;
	double runtime = 0;
	double memtime = 0;
	int i=0;
	while(i<iteration){
		t1 = getTime();
		nbody.update(activeParams.m_timestep);
		t2 = getTime();
		runtime += t2-t1;
		i++;
		if(i%outInterval ==0){
			t1 = getTime();
			int offset = (i/outInterval)*numBodies*4;
			memcpy(body_outBuffer+offset, nbody.getPosArray(), numBodies*4*sizeof(FTYPE) );
			t2 = getTime();
			memtime += t2-t1;
		}
	}

	/*
	 * output to a file
	 */
	if(outfilename){
		FILE *fp = fopen(outfilename, "w");
		if(!fp){
			std::cout<<"Cann't open the output file !!!"<<std::endl;
			exit(1);
		}
		for(int i=0; i<iteration/outInterval + 1; i++){
			fprintf(fp, "%f\n", i*activeParams.m_timestep);
			for(int j=0; j<numBodies; j++){
				fprintf(fp, "%.5f ", body_outBuffer[i*numBodies*4 + j*4 + 0]);
				fprintf(fp, "%.5f ", body_outBuffer[i*numBodies*4 + j*4 +1]);
				fprintf(fp, "%.5f ", body_outBuffer[i*numBodies*4 + j*4 +2]);
				fprintf(fp, "%.5f\n", body_outBuffer[i*numBodies*4 + j*4 +3]);
			}
		}
		fclose(fp);
	}

	delete[] body_pos;
	delete[] body_vel;
	delete[] body_color;
	delete[] body_outBuffer;

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tbody system info:"<<std::endl;
		std::cout<<"\t\tTotal bodies : "<<numBodies<<std::endl;
		std::cout<<"\t\tIterations: "<<iteration<<std::endl;
		std::cout<<"\t\tOutput frames: "<<iteration/outInterval + 1<<std::endl;
		std::cout<<"\ttime info: "<<std::endl;
		std::cout<<"\t\truntime: "<<std::fixed<<std::setprecision(4)<<1000*runtime<<"(ms)"<<std::endl;
		std::cout<<"\t\tmemtime: "<<std::fixed<<std::setprecision(4)<<1000*memtime<<"(ms)"<<std::endl;
	}

	double totaltime = runtime + memtime;
	totaltime *= 1000;
	print_time(1, &totaltime);

	return 0;

}









