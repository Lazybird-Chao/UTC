/*
 *
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

class FileWriter: public UserTaskBase{
private:
	char filename[20];
	int iter;
	int dimx, dimy;
	double *buffer;

	int nproc;
	int partmx;
	int leftmx;

	Conduit* dataCdt;

public:
	void initImpl(int iter, int x, int y, int nproc, Conduit* cdt){
		if(getUniqueExecution()){
			dimx = x;
			dimy = y;
			this->iter = iter;
			buffer = (double*)malloc((dimx+2)*(dimy+2)*sizeof(double));
			dataCdt = cdt;
			this->nproc = nproc;

			partmx = (dimx + __numProcesses -1) / __numProcesses;
			leftmx = dimx - partmx *(__numProcesses -1);
			partmx +=2;
		}

		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl on "
					<<__processId<<".\n";
		}
	}

	void runImpl(double *runtime){
		DIM2 (double, re, dimy+2);
		Timer timer;
		//timer.start();
		for(int i=0; i<iter; i++){
			char fidx[2];
			fidx[1]='\0';
			fidx[0]= '0'+i;
			char filename[20];
			strcpy(filename,"./figure/");
			strcat(filename,fidx);
			strcat(filename,".img");
			FILE* fp = fopen(filename, "w");

			dataCdt->Read(buffer, (dimx+2)*(dimy+2)*sizeof(double), i);

			timer.start();
			for(int i=0; i<__numProcesses-1; i++){
				re = (double (*)[dimy+2])(&buffer[i*partmx*(dimy+2)]);
				for(int j=1; j<(dimx+1); j++)
					fwrite(&(re[j][1]), dimy, sizeof(double), fp);
			}
			re = (double (*)[dimy+2])(buffer+(__numProcesses-1)*partmx*(dimy+2));

			for(int j=1; j<leftmx+1; j++){
				fwrite(&(re[j][1]), dimy , sizeof(double), fp);
			}
			fclose(fp);
			runtime[__localThreadId] += timer.stop();
		}


		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl on "<<
					__processId<<".\n";
		}
	}

	~FileWriter(){
		if(buffer)
			free(buffer);
	}
};


class HeatImage: public UserTaskBase{
private:
	int mx;
	int my;
	int partmx;
	int leftmx;
	int totalmx;
	double rdx2, rdy2, beta;
	int NITER;

	int Nfiles;
	int dimx, dimy;

	GlobalScopedData<double> *doublef;
	//double *f;
	//double *newf;
	double *r;

	Conduit* cdt;



public:
	void initImpl(int dimx, int dimy, int iters, int nfiles, Conduit* cdt){
		if(getUniqueExecution()){
			this->dimx = dimx;
			this->dimy = dimy;
			NITER = iters;
			Nfiles = nfiles;
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

			this->cdt = cdt;
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
		int linesperthread = (mx-2)/__numLocalThreads;
		int residue = (mx-2)%__numLocalThreads;
		int starti, endi;
		starti = __localThreadId*linesperthread;
		endi = starti+linesperthread;
		if(__localThreadId == __numLocalThreads-1)
			endi = mx-2;
		//std::cout<<starti<<" "<<endi<<std::endl;
		timer.start();
		for(int m =0; m< Nfiles; m++ ){
			if(__globalThreadId == 0){
				std::cout<<"Iteration "<<m<<std::endl;
			}
			timer1.start();
		for( int n=0; n< NITER; n++){
			/*if(__globalThreadId == 0){
				if(n%STEPITER == 0)
					std::cout<<"Iteration "<<n<<std::endl;
			}*/
			//timer1.start();
			f = pf[curf];
			newf = pf[1-curf];
			/*computing */
			timer2.start();
			for (int i = 1 + starti; i < 1 + endi; i++) {
				for (int j = 1; j < my-1; j++) {
					newf[i][j] =
						((f[i - 1][j] + f[i + 1][j]) * rdx2 +
						 (f[i][j - 1] + f[i][j + 1]) * rdy2 - r[i*my + j]) * beta;
				}
			}
			//runtime[__localThreadId*3+1] += timer2.stop();
			inter_Barrier();

			/* exchanging bondary data */
			//timer2.start();
			if(__numProcesses>1){
				if(getUniqueExecution()){
					if(__processId>0){
						doublef->rstoreblock(__processId-1, &(newf[1][1]), basef[1-curf]+(partmx-1)*my+1, my-2);
						//doublef->rstoreQuiet();
						//doublef->rstoreSetFinishFlag(__processId-1);
					}
					if(__processId < __numProcesses -1){
						doublef->rstoreblock(__processId+1, &(newf[mx-2][1]), basef[1-curf]+1, my-2);
						//doublef->rstoreQuiet();
						//doublef->rstoreSetFinishFlag(__processId+1);
					}
					shmem_barrier_all();
					/*
					//doublef->rstoreQuiet();
					if(__processId>0)
						doublef->rstoreWaitFinishFlag(__processId-1);
					if(__processId<__numProcesses -1)
						doublef->rstoreWaitFinishFlag(__processId+1);
					*/
					/*if(__processId>0)
						doublef->rloadblock(__processId-1, &(newf[0][1]), basef[1-curf]+(mx-2)*my+1, my-2);
					if(__processId<__numProcesses-1)
						doublef->rloadblock(__processId+1, &(newf[partmx-1][1]), basef[1-curf]+(1*my+1), my-2);
						*/

				}
				//intra_Barrier();
			}
			curf = 1-curf;
			//runtime[__localThreadId] += timer1.stop();
		}
		inter_Barrier();

		// gather all results to one process
		double * results;
		if(__globalThreadId==0){
			results = (double*)malloc(partmx*my*__numProcesses*sizeof(double));
		}
		//std::cout<<results<<std::endl;
		SharedDataGather(&newf[0][0], sizeof(double)*partmx*my, results, 0);
		cdt->WriteBy(0,results,partmx*my*__numProcesses*sizeof(double),m );
		intra_Barrier();
		if(__globalThreadId==0)
			free(results);
		runtime[__localThreadId] += timer1.stop();
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
	std::cout<<"here"<<std::endl;
	int nthreads=0;
	int nprocs=0;
	int dimx=600;
	int dimy=800;
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

	double *runtime1 = new double[nthreads];
	double *runtime2 = new double[1];
	for(int i=0; i<nthreads; i++)
		runtime1[i]=0;
	runtime2[0]=0;
	ProcList plist1;
	for(int i=0; i<nprocs; i++)
		for(int j=0; j<nthreads; j++)
			plist1.push_back(i);

	Task<HeatImage> heatImageInst(plist1);

	Task<FileWriter> filewriter(ProcList(0));

	Conduit cdt(&heatImageInst, &filewriter);

	heatImageInst.init(dimx, dimy, iters, 10, &cdt);
	filewriter.init(10,dimx,dimy,nprocs, &cdt);
	Timer timer;
	timer.start();
	heatImageInst.run(runtime1);
	filewriter.run(runtime2);
	heatImageInst.wait();
	filewriter.wait();
	double totaltime = timer.stop();

	double avg_runtime1=0;
	double avg_runtime2=0;
	double write_time=0;
	for(int i =0; i<nthreads; i++){
		avg_runtime1+= runtime1[i];
	}
	avg_runtime1/=nthreads;
	write_time = runtime2[0];
	ctx.Barrier();
	MPI_Reduce(&avg_runtime1, &avg_runtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avg_runtime2/=nprocs;

	if(myproc==0){
		std::cout<<"total time: "<<totaltime<<std::endl<<
				"compute time: "<<avg_runtime1<<std::endl<<
					"\twrite time: "<<write_time<<std::endl;
		double timer[3];
		timer[0] = totaltime;
		timer[1] = avg_runtime1;
		timer[2] = write_time;
		print_time(3,timer);
	}
	delete runtime1;
	delete runtime2;

	return 0;
}
