/*
 * utc_shheat_image_v2.cc
 *
 *  Created on: Jul 21, 2016
 *      Author: chao
 */


#include "Utc.h"
#include "../helper_getopt.h"
#include "../helper_printtime.h"

#include <iostream>

#define DIM2(basetype, name, w1) basetype (*name)[w1]
#define DIM3(basetype, name, w1, w2) basetype (*name)[w1][w2]
#define STEPITER 1000
#define delx 0.5
#define dely 0.25

using namespace iUtc;
class HeatImage: public UserTaskBase{
private:
	int mx;
	int my;
	int partmx;
	int leftmx;
	int totalmx;
	double rdx2, rdy2, beta;
	int NITER;

	GlobalScopedData<double> *doublef;
	//double *f;
	//double *newf;
	double *r;

public:
	void initImpl(int dimx, int dimy, int iters){
		if(getUniqueExecution()){
			NITER = iters;
			this->mx = dimx;
			this->my = dimy;
			totalmx = mx;
			this->mx = (totalmx + __numProcesses -1) / __numProcesses;
			/* This is the number of rows for all but the last: */
			partmx = this->mx;
			/* This is the number of rows for the last: */
			leftmx = totalmx - partmx *(__numProcesses -1);
			if(leftmx <1){
				std::cerr<<"Cannot distribute rows, too many processors\n";
				exit(-1);
			}
			if(__processId == __numProcesses-1)
				this->mx = leftmx;
			partmx += 2;
			this->mx += 2;
			this->my += 2;

			doublef = new GlobalScopedData<double>(2*partmx*this->my);
			r = (double*)malloc(this->mx*this->my*sizeof(double));
			rdx2 = 1./delx/delx;
			rdy2 = 1./dely/dely;
			beta = 1.0/(2.0*(rdx2+rdy2));
			std::cout<<"Solving heat conduction task on "<< totalmx <<" by "<<this->my-2
					<<" grid by "<<__processId<<" processors\n";

			//initialize data
			DIM2 (double, f, this->my) = (double(*)[this->my])doublef->getPtr();
			DIM2 (double, newf, this->my) = f + partmx;
			for(int i=0; i< this->mx; i++)
				for(int j =0; j<this->my; j++){
					if(((i == 0) && (__processId == 0)) || (j == 0)
						|| ((i == (this->mx - 1)) && (__processId == (__numProcesses - 1)))
						|| (j == (this->my - 1))){
						newf[i][j] = f[i][j] = 1.0;
					}
					else{
						newf[i][j] = f[i][j] = 0.0;
					}
					r[i*this->my + j]=0.0;
				}
		}

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl on "
					<<__processId<<".\n";
		}

	}

	void runImpl(double *runtime){
		Timer timer, timer1, timer2;
		inter_Barrier();
		timer.start();

		DIM2 (double, f, my);
		DIM2 (double, newf, my);
		double (*pf[2])[my];
		int curf=0;
		int basef[2];
		pf[0] = (double(*)[my])doublef->getPtr();
		basef[0]= 0;
		basef[1]= partmx*my;
		pf[1] = pf[0] + partmx;
		//std::cout<<pf[0]<<" "<<pf[0]+1<<std::endl;

		/* iteration for computing */
		int SPEC_ThreadId = __numLocalThreads-1; // let the last local thread to be the special thread
		int NLINES = 0 + 2;  // how many lines the special thread response.
		int linesperthread;
		int residue;
		int starti, endi;
		if(__numLocalThreads<2){
			starti = 1;
			endi = mx-2;
		}
		else{
			if(__numProcesses <2){
				linesperthread = (mx-2)/(__numLocalThreads);
				residue = (mx-2)%(__numLocalThreads);
				starti = __localThreadId*linesperthread + 1;
				endi = starti+linesperthread-1;
				if(__localThreadId == __numLocalThreads-1)
					endi = mx-2;
			}
			else{
				linesperthread = (mx-2-NLINES)/(__numLocalThreads-1);
				residue = (mx-2-NLINES)%(__numLocalThreads-1);
				starti = __localThreadId*linesperthread + NLINES;
				endi = starti+linesperthread-1;
				if(__localThreadId == __numLocalThreads-2)
					endi = mx-2 -1;
			}
		}

		for( int n=0; n< NITER; n++){
			if(__globalThreadId == 0){
				if(n%STEPITER == 0)
					std::cout<<"Iteration "<<n<<std::endl;
			}
			timer1.start();
			f = pf[curf];
			newf = pf[1-curf];
			//std::cout<<ERROR_LINE<<std::endl;
			/*computing */
			if(__numLocalThreads<2){
				//only one thread in a process
				for (int i = starti; i <=endi; i++) {
					for (int j = 1; j < my-1; j++) {
						newf[i][j] =
							((f[i - 1][j] + f[i + 1][j]) * rdx2 +
							 (f[i][j - 1] + f[i][j + 1]) * rdy2 - r[i*my + j]) * beta;
					}
				}
				inter_Barrier();

				timer2.start();
				if(__numProcesses>1){
					if(__processId>0){
						doublef->rstoreblock(__processId-1, &(newf[1][1]), basef[1-curf]+(partmx-1)*my+1, my-2);
						//doublef->rstoreQuiet();
						doublef->rstoreSetFinishFlag(__processId-1);
					}
					if(__processId < __numProcesses -1){
						doublef->rstoreblock(__processId+1, &(newf[mx-2][1]), basef[1-curf]+1, my-2);
						//doublef->rstoreQuiet();
						doublef->rstoreSetFinishFlag(__processId+1);
					}
					//doublef->rstoreQuiet();
					if(__processId>0)
						doublef->rstoreWaitFinishFlag(__processId-1);
					if(__processId<__numProcesses -1)
						doublef->rstoreWaitFinishFlag(__processId+1);

					/*if(__processId>0)
						doublef->rloadblock(__processId-1, &(newf[0][1]), basef[1-curf]+(mx-2)*my+1, my-2);
					if(__processId<__numProcesses-1)
						doublef->rloadblock(__processId+1, &(newf[partmx-1][1]), basef[1-curf]+(1*my+1), my-2);
						*/
				}
				curf = 1-curf;
				runtime[__localThreadId*3+1] += timer2.stop();
				runtime[__localThreadId*3] += timer1.stop();

			}
			else{
				//std::cout<<"("<<__processId<<" "<<__localThreadId<<" "<<n<<") ";
				//inter_Barrier();
				timer2.start();
				//more than 2 threads, we use one thread for comm and do less compute
				if(__numProcesses>1 && __localThreadId == SPEC_ThreadId){

					// the special thread (last thread)
					for (int j = 1; j < my-1; j++) {
						// i=1;
						newf[1][j] =
							((f[0][j] + f[2][j]) * rdx2 +
							 (f[1][j - 1] + f[1][j + 1]) * rdy2 - r[my + j]) * beta;
						// i=mx-2
						newf[mx-2][j] =
							((f[mx-3][j] + f[mx-1][j]) * rdx2 +
							 (f[mx-2][j - 1] + f[mx-2][j + 1]) * rdy2 - r[(mx-2)*my + j]) * beta;
					}
					shmem_barrier_all();
					if(__processId>0)
						doublef->rstoreblock(__processId-1, &(newf[1][1]), basef[1-curf]+(partmx-1)*my+1, my-2);
					if(__processId<__numProcesses-1)
						doublef->rstoreblock(__processId+1, &(newf[mx-2][1]), basef[1-curf]+1, my-2);
					shmem_barrier_all();
					/*
					if(__processId>0)
						doublef->rstoreSetFinishFlag(__processId-1);
					if(__processId<__numProcesses-1)
						doublef->rstoreSetFinishFlag(__processId+1);
					*/
					for (int i = 2; i <= NLINES-1; i++) {
						for (int j = 1; j < my-1; j++) {
							newf[i][j] =
								((f[i - 1][j] + f[i + 1][j]) * rdx2 +
								 (f[i][j - 1] + f[i][j + 1]) * rdy2 - r[i*my + j]) * beta;
						}
					}
					/*
					if(__processId>0){
						doublef->rstoreWaitFinishFlag(__processId-1);
						doublef->rloadblock(__processId-1, &(newf[0][1]), basef[1-curf]+(mx-2)*my+1, my-2);
					}
					if(__processId<__numProcesses-1){
						doublef->rstoreWaitFinishFlag(__processId+1);
						doublef->rloadblock(__processId+1, &(newf[partmx-1][1]), basef[1-curf]+(1*my+1), my-2);
					}*/

				}
				else{
					// other threads
					for (int i = starti; i <=endi; i++) {
						for (int j = 1; j < my-1; j++) {
							newf[i][j] =
								((f[i - 1][j] + f[i + 1][j]) * rdx2 +
								 (f[i][j - 1] + f[i][j + 1]) * rdy2 - r[i*my + j]) * beta;
						}
					}

					//std::cout<<"("<<__processId<<" "<<__localThreadId<<" "<<n<<") ";
				}
				runtime[__localThreadId*3+2] += timer2.stop();
				curf = 1-curf;
				//intra_Barrier();
				inter_Barrier();
				runtime[__localThreadId*3] += timer1.stop();
			}

		}

		char outfile[20] = "out.img";
		if (!__globalThreadId) {
			printf ("Elapsed time: %4.2f sec\n", timer.stop());
			printf ("Output image file in current directory\n");
			FILE*fp = fopen (outfile, "w");
			fclose (fp);
		}


		for (int j = 0; j < __numProcesses; j++) {
			inter_Barrier();
			if(getUniqueExecution()){
				if (j == __processId) {
					FILE* fp = fopen (outfile, "a");
					for (int i = 1; i < (mx - 1); i++)
						fwrite (&(newf[i][1]), my - 2, sizeof (newf[0][0]), fp);
					fclose (fp);
				}
			}
		}

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl on "<<
					__processId<<".\n";
		}

	}

	~HeatImage(){
		if(doublef)
			delete doublef;
		if(r)
			free(r);
	}
};

int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads;
	int nprocs;
	int dimx;
	int dimy;
	int iters=0;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:x:y:i:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case 'x':
			dimx = atoi(optarg);
			break;
		case 'y':
			dimy = atoi(optarg);
			break;
		case 'i':
			iters = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:x:y:i:");
	}
	if(iters==0)
		iters = 5000;
	//std::cout<<iters<<std::endl;
	int procs = ctx.numProcs();
	if(nprocs != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}
	int myproc = ctx.getProcRank();

	double *runtime = new double[3*nthreads];
	for(int i=0; i<3*nthreads; i++)
		runtime[i]=0;
	ProcList plist1;
	for(int i=0; i<nprocs; i++)
		for(int j=0; j<nthreads; j++)
			plist1.push_back(i);
	Task<HeatImage> heatImageInst(plist1);
	heatImageInst.init(dimx, dimy, iters);
	heatImageInst.run(runtime);
	heatImageInst.wait();

	if(nthreads <2){
		double avg_comptime1=0;
		double avg_comptime2=0;
		double avg_commtime1=0;
		double avg_commtime2=0;

		avg_comptime1+= runtime[0]-runtime[1];
		avg_commtime1+= runtime[1];
		std::cout<<myproc<<": "<<avg_comptime1<<" "<<avg_commtime1<<std::endl;
		ctx.Barrier();
		MPI_Reduce(&avg_comptime1, &avg_comptime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		avg_comptime2/=nprocs;
		MPI_Reduce(&avg_commtime1, &avg_commtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		avg_commtime2/=nprocs;
		if(myproc==0)
			std::cout<<"average run() compute time: "<<avg_comptime2<<std::endl<<
						"\tcommunicate time: "<<avg_commtime2<<std::endl;
	}
	else{
		double avg_runtime1=0;
		double avg_runtime2=0;
		double firsttime=0;
		double lasttime=0;
		for(int i=0; i<nthreads; i++){
			avg_runtime1 += runtime[i*3];

		}
		avg_runtime1 /= nthreads;
		for(int i=0; i<nthreads-1; i++)
			firsttime +=runtime[i*3+2];
		firsttime /= (nthreads-1);

		//std::cout<<myproc<<": "<<avg_runtime1<<std::endl;
		ctx.Barrier();
		MPI_Reduce(&avg_runtime1, &avg_runtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		avg_runtime2 /= nprocs;
		if(myproc==0){
			std::cout<<"average run() time: "<<avg_runtime2<<std::endl<<
					"\tlast thread time: "<<runtime[(nthreads-1)*3+2]<<std::endl<<
					"\tfirst thread time: "<<firsttime<<std::endl;
			double timer[3];
			timer[0] = avg_runtime1;
			timer[1] = runtime[(nthreads-1)*3+2];
			timer[2] = firsttime;
			print_time(3,timer);
		}
		/*
		ctx.Barrier();
		for(int i=0;i<nprocs; i++){
			if(myproc == i){
				std::cout<<"run() time of each thread on "<<i<<": \n";
				for(int j=0; j<nthreads;j++)
					std::cout<<"\t"<<j<<": "<<runtime[2*j+1]<<std::endl;
			}
			ctx.Barrier();
		}
		*/

	}
	delete runtime;

	return 0;
}




