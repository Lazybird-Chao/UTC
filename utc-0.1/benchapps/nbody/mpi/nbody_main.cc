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
#include "mpi.h"

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "nbody.h"
#include "bodysystem.h"

#define FTYPE double
#define MPI_FTYPE MPI_DOUBLE

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
	int nprocess = 1;

	/*
	 * parse arguments
	 */
	int opt;
	extern char* optarg;
	extern int optind;
	while((opt=getopt(argc, argv, "vi:l:o:O:n:p:"))!=EOF){
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
		case 'p':
			nprocess = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
	}

	MPI_Init(&argc, &argv);
	int procs;
	int myproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
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
	FTYPE *body_outBuffer;
	if(myproc == 0){
		randomizeBodies(NBODY_CONFIG_SHELL, body_pos, body_vel, body_color,
				activeParams.m_clusterScale, activeParams.m_velocityScale,
				numBodies, true);
		body_outBuffer = new FTYPE[numBodies*4*(iteration/outInterval + 1)];
		// copy the initial pos info to out
		memcpy(body_outBuffer, body_pos, numBodies*4*sizeof(FTYPE));
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t1, t2;
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;
	/*
	 * bcast init body and vel to all nodes
	 */
	t1 = MPI_Wtime();
	t2 = t1;
	MPI_Bcast(body_pos, numBodies*4, MPI_FTYPE, 0, MPI_COMM_WORLD);
	MPI_Bcast(body_vel, numBodies*4, MPI_FTYPE, 0, MPI_COMM_WORLD);
	commtime += MPI_Wtime() - t2;

	/*
	 * create bodysystem
	 */
	int blockBodies = numBodies / nprocess;
	int startBodyIndex = blockBodies * myproc;
	BodySystem<FTYPE> nbody(numBodies, blockBodies, startBodyIndex);
	nbody.setDamping(activeParams.m_damping);
	nbody.setSoftening(activeParams.m_softening);
	nbody.setPosArray(body_pos);
	nbody.setVelArray(body_vel);

	/*
	 * iterate body system
	 */
	if(myproc == 0)
		std::cout<<"start iteration ..."<<std::endl;
	int i=0;
	FTYPE *pos = nbody.getPosArray();
	FTYPE *new_pos = nbody.getNewPosArray();
	while(i<iteration){
		t2 = MPI_Wtime();
		nbody.update(activeParams.m_timestep);
		computetime += MPI_Wtime() - t2;

		/*
		 * gather new bodies from all nodes
		 */
		t2 = MPI_Wtime();
		MPI_Gather(new_pos, blockBodies*4, MPI_FTYPE,
				pos, blockBodies*4, MPI_FTYPE,
				0, MPI_COMM_WORLD);
		MPI_Bcast(pos, numBodies*4, MPI_FTYPE, 0, MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t2;

		i++;
		if(i%outInterval ==0 && myproc ==  0){
			int offset = (i/outInterval)*numBodies*4;
			memcpy(body_outBuffer+offset, pos, numBodies*4*sizeof(FTYPE) );

		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	totaltime = MPI_Wtime() - t1;
	/*
	 * output to a file
	 */
	if(outfilename && myproc == 0){
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
	if(myproc == 0)
		delete[] body_outBuffer;

	double runtime[3];
	MPI_Reduce(&totaltime, runtime+0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&computetime, runtime+1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&commtime, runtime+2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(myproc == 0){
		for(int i = 0; i< 3; i++)
			runtime[i] /= nprocess;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tbody system info:"<<std::endl;
			std::cout<<"\t\tTotal bodies : "<<numBodies<<std::endl;
			std::cout<<"\t\tIterations: "<<iteration<<std::endl;
			std::cout<<"\t\tOutput frames: "<<iteration/outInterval + 1<<std::endl;
			std::cout<<"\ttime info: "<<std::endl;
			std::cout<<"\t\ttotal time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[0]<<"(ms)"<<std::endl;
			std::cout<<"\t\tcompute time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[1]<<"(ms)"<<std::endl;
			std::cout<<"\t\tcomm time: "<<std::fixed<<std::setprecision(4)<<1000*runtime[2]<<"(ms)"<<std::endl;
		}

		for(int i = 0; i< 3; i++)
			runtime[i] *=1000;
		print_time(3, runtime);
	}
	return 0;

}









