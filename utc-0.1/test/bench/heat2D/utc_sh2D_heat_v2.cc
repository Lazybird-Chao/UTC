/*
 *
 */

#include "Utc.h"
#include "../helper_getopt.h"
#include "../helper_printtime.h"

#include <iostream>

#define H 1.0
#define _EPSILON 0.1

#define METHOD 1
#define T_SRC0 550.0
#define ROOT 0

using namespace iUtc;
class Heat2D: public UserTaskBase{
private:
	int width;
	int heigth;
	int method;
	float EPSILON = _EPSILON;
	float convergence;
	float convergence_sqd, local_convergence_sqd;
	float *local_convergence_sqd_array;

	int my_start_row;
	int my_end_row;
	int my_num_rows;

	PrivateScopedData<int> t_start_row;
	PrivateScopedData<int> t_end_row;
	PrivateScopedData<int> t_num_rows;

	float *U_Curr;
	float *U_Next;
	GlobalScopedData<float> *U_Curr_Above;
	GlobalScopedData<float> *U_Curr_Below;
	GlobalScopedData<float> *U_Above_Buffer;
	GlobalScopedData<float> *U_Below_Buffer;

	int get_start(int rank){
		/* computer row divisions to each proc */
		int per_proc, start_row, remainder;
		// MPI_Comm_size(MPI_COMM_WORLD,&p);

		/* get initial whole divisor */
		per_proc = (int) floor (heigth / H) / __numProcesses;
		/* get number of remaining */
		remainder = (int) floor (heigth / H) % __numProcesses;
		/* there is a remainder, then it distribute it to the first "remainder"
		   procs */
		if (rank < remainder) {
			start_row = rank * (per_proc + 1);
		}
		else {
			start_row = rank * (per_proc) + remainder;
		}
		return start_row;
	}

	int get_end(int rank ){
		 /* computer row divisions to each proc */
		int per_proc, remainder, end_row;
		// MPI_Comm_size(MPI_COMM_WORLD,&p);
		per_proc = (int) floor (heigth / H) / __numProcesses;
		remainder = (int) floor (heigth / H) % __numProcesses;
		if (rank < remainder) {
			end_row = get_start (rank) + per_proc;
		}
		else {
			end_row = get_start (rank) + per_proc - 1;
		}
		return end_row;
	}

	int get_num_rows(int rank){
		return 1+ get_end(rank) - get_start(rank);
	}

	int global_to_local(int rank, int row){
		return row - get_start(rank);
	}

	void init_domain(float *domain_ptr){
		float(*p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])domain_ptr;
		for (int j = my_start_row; j <= my_end_row; j++) {
			for (int i = 0; i < (int) floor (width / H); i++) {
				p[j - my_start_row][i] = 0.0;
			}
		}
	}

	void get_convergence_sqd(float *current_ptr, float *next_ptr ){
		float sum = 0;
		float(*c_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])current_ptr;
		float(*n_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])next_ptr;
		/*if(getUniqueExecution()){
			for (int j = my_start_row; j <= my_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					sum +=
						pow (n_p[global_to_local (__processId, j)][i] -
							 c_p[global_to_local (__processId, j)][i], 2);
				}
			}
			local_convergence_sqd= sum;
			//std::cout<<local_convergence_sqd<<std::endl;
		}*/
		for (int j = t_start_row; j <= t_end_row; j++) {
			for (int i = 0; i < (int) floor (width / H); i++) {
				sum +=
					pow (n_p[j-my_start_row][i] -
						 c_p[j-my_start_row][i], 2);
			}
		}

		local_convergence_sqd_array[__localThreadId]= sum;
		intra_Barrier();
		if(getUniqueExecution()){
			local_convergence_sqd = 0;
			for(int i =0; i< __numLocalThreads; i++)
				local_convergence_sqd += local_convergence_sqd_array[i];
		}
		intra_Barrier();

	}

	void enforce_bc_par(float *domain_ptr, int rank, int i, int j){
		//float(*p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])domain_ptr;
		/* enforce bc's first */
		if (i == ((int) floor (width / H / 2) - 1) && j == 0) {
			/* This is the heat source location */
			domain_ptr[j* ((int) floor (width / H)) +i] = T_SRC0;
		}
		else if (i <= 0 || j <= 0 || i >= ((int) floor (width / H) - 1)
				 || j >= ((int) floor (width / H) - 1)) {
			/* All edges and beyond are set to 0.0 */
			domain_ptr[(j-my_start_row)*((int) floor (width / H)) +i] = 0.0;
		}
	}

	float get_val_par (float *above_ptr, float *domain_ptr, float *below_ptr, int rank,
	             int i, int j)
	{
	    float ret_val;

	    /* enforce bc's first */
	    if (i == ((int) floor (width / H / 2) - 1) && j == 0) {
	        /* This is the heat source location */
	        ret_val = T_SRC0;
	    }
	    else if (i <= 0 || j <= 0 || i >= ((int) floor (width / H) - 1)
	             || j >= ((int) floor (heigth / H) - 1)) {
	        /* All edges and beyond are set to 0.0 */
	        ret_val = 0.0;
	    }
	    else {
	        /* Else, return value for matrix supplied or ghost rows */
	        if (j < my_start_row) {
	            if (rank == ROOT) {
	                /* not interested in above ghost row */
	                ret_val = 0.0;
	            }
	            else {
	                ret_val = above_ptr[i];
	                /* printf("%d: Used ghost (%d,%d) row from above =
	                   %f\n",rank,i,j,above_ptr[i]); fflush(stdout); */
	            }
	        }
	        else if (j > my_end_row) {
	            if (rank == (__numProcesses - 1)) {
	                /* not interested in below ghost row */
	                ret_val = 0.0;
	            }
	            else {
	                ret_val = below_ptr[i];
	                /* printf("%d: Used ghost (%d,%d) row from below =
	                   %f\n",rank,i,j,below_ptr[i]); fflush(stdout); */
	            }
	        }
	        else {
	            /* else, return the value in the domain asked for */
	            ret_val = domain_ptr[(j-my_start_row)*((int) floor (width / H)) + i];
	            /* printf("%d: Used real (%d,%d) row from self =
	               %f\n",rank,i,global_to_local(rank,j),domain_ptr[global_to_local(rank,j)][i]);
	               fflush(stdout); */
	        }
	    }
	    return ret_val;

	}

	float f (int i, int j)
	{
	    return 0.0;
	}

	void jacobi(float *current_ptr, float *next_ptr, double *runtime, int t_s, int t_e, int NLINES){
		float(*c_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])current_ptr;
		float(*n_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])next_ptr;
		Timer timer;
		/* transfer bondary data*/
		if(__numLocalThreads<2 || __numProcesses <2){
			timer.start();
			if(__numProcesses>1){
				if(getUniqueExecution()){
					if(__processId < __numProcesses-1){
						U_Curr_Above->rstoreblock(__processId+1, &c_p[my_num_rows-1][0], 0, (int) floor (width / H));
						//U_Curr_Above->rstoreSetFinishFlag(__processId+1);
					}
					shmem_barrier_all();
					if(__processId > 0){
						U_Curr_Below->rstoreblock(__processId-1, &c_p[0][0], 0, (int) floor (width / H));
						//U_Curr_Below->rstoreSetFinishFlag(__processId-1);
					}
					shmem_barrier_all();
					/*if(__processId<__numProcesses-1)
						U_Curr_Below->rstoreWaitFinishFlag(__processId+1);
					if(__processId>0)
						U_Curr_Above->rstoreWaitFinishFlag(__processId-1);
						*/
				}
				inter_Barrier();
			}
			runtime[4*__localThreadId+2] += timer.stop();
			timer.start();
			for (int j = t_start_row; j <= t_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					n_p[j - my_start_row][i] =
						.25 * (
						get_val_par(U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId, i - 1,j)
					+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(),__processId, i + 1, j)
					+ get_val_par (U_Curr_Above->getPtr(), current_ptr,U_Curr_Below->getPtr(), __processId, i, j - 1)
					+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId,i, j + 1)
					- (pow (H, 2) * f (i, j))
					);
					enforce_bc_par (next_ptr, __processId, i, j);
				}
			}
			runtime[4*__localThreadId+1] += timer.stop();
			intra_Barrier();
		}
		else{ // more than 1 local threads
			timer.start();
			if(__localThreadId == __numLocalThreads-1){
				// last thread as special
				if(__processId<__numProcesses -1){
					U_Curr_Above->rstoreblock(__processId+1, &c_p[my_num_rows-1][0], 0, (int) floor (width / H));
					//U_Curr_Above->rstoreSetFinishFlag(__processId+1);
				}
				if(__processId>0){
					U_Curr_Below->rstoreblock(__processId-1, &c_p[0][0], 0, (int) floor (width / H));
					//U_Curr_Below->rstoreSetFinishFlag(__processId-1);
				}
				for(int j=my_start_row+1; j<=my_start_row+NLINES-2; j++){
					for (int i = 0; i < (int) floor (width / H); i++) {
						n_p[j - my_start_row][i] =
							.25 * (
							get_val_par(U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId, i - 1,j)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(),__processId, i + 1, j)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr,U_Curr_Below->getPtr(), __processId, i, j - 1)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId,i, j + 1)
						- (pow (H, 2) * f (i, j))
						);
						enforce_bc_par (next_ptr, __processId, i, j);
					}
				}
				/*if(__processId<__numProcesses-1)
					U_Curr_Below->rstoreWaitFinishFlag(__processId+1);
				if(__processId>0)
					U_Curr_Above->rstoreWaitFinishFlag(__processId-1);*/
				shmem_barrier_all();
				for (int i = 0; i < (int) floor (width / H); i++) {
					n_p[my_start_row - my_start_row][i] =
							.25 * (
							get_val_par(U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId, i - 1,my_start_row)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(),__processId, i + 1, my_start_row)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr,U_Curr_Below->getPtr(), __processId, i, my_start_row - 1)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId,i, my_start_row + 1)
						- (pow (H, 2) * f (i, my_start_row))
						);
					enforce_bc_par (next_ptr, __processId, i, t_start_row);
					n_p[my_end_row - my_start_row][i] =
							.25 * (
							get_val_par(U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId, i - 1,my_end_row)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(),__processId, i + 1, my_end_row)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr,U_Curr_Below->getPtr(), __processId, i, my_end_row - 1)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId,i, my_end_row + 1)
						- (pow (H, 2) * f (i, t_end_row))
						);
					enforce_bc_par (next_ptr, __processId, i, my_end_row);
				}
			}
			else{
				// other threads
				for (int j = t_s; j <= t_e; j++) {
					for (int i = 0; i < (int) floor (width / H); i++) {
						n_p[j - my_start_row][i] =
							.25 * (
							get_val_par(U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId, i - 1,j)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(),__processId, i + 1, j)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr,U_Curr_Below->getPtr(), __processId, i, j - 1)
						+ get_val_par (U_Curr_Above->getPtr(), current_ptr, U_Curr_Below->getPtr(), __processId,i, j + 1)
						- (pow (H, 2) * f (i, j))
						);
						enforce_bc_par (next_ptr, __processId, i, j);
					}
				}
			}
			runtime[4*__localThreadId+2] += timer.stop();
			intra_Barrier();
			runtime[4*__localThreadId+1] += timer.stop();
		}

	}

	void gauss_seidel(float *current_ptr, float *next_ptr){

	}

	void sor(float *current_ptr, float *next_ptr){

	}

public:
	void initImpl(int w, int h, int meth){
		if(getUniqueExecution()){
			width = w;
			heigth = h;
			method = meth;

			my_start_row = get_start(__processId);
			my_end_row = get_end(__processId);
			my_num_rows = get_num_rows(__processId);

			U_Curr = (float *) malloc (sizeof (float) * my_num_rows *
                    				(int) floor (width / H));
			U_Next = (float *) malloc (sizeof (float) * my_num_rows *
    								(int) floor (width / H));
			init_domain(U_Curr);
			init_domain(U_Next);

			U_Curr_Above = new GlobalScopedData<float>((int) floor (width / H));
			U_Curr_Below = new GlobalScopedData<float>((int) floor (width / H));
			U_Above_Buffer = new GlobalScopedData<float>((int) floor (width / H));
			U_Below_Buffer = new GlobalScopedData<float>((int) floor (width / H));

			local_convergence_sqd_array = new float[__numLocalThreads];
		}
		inter_Barrier();

		int rows_per_thread = my_num_rows / __numLocalThreads;
		t_start_row = __localThreadId * rows_per_thread + my_start_row;
		t_end_row = t_start_row + rows_per_thread -1 ;
		t_num_rows = rows_per_thread;
		if(__localThreadId == __numLocalThreads-1){
			t_end_row = my_end_row;
			t_num_rows = my_num_rows - (__numLocalThreads-1)*rows_per_thread;
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

		int t_s, t_e, NLINES;
		if(__numLocalThreads>1 &&__numProcesses>1){
			NLINES = 0 + 2;
			int rows_per_thread = (my_num_rows - NLINES) / (__numLocalThreads-1);
			int residue = (my_num_rows - NLINES) % (__numLocalThreads-1);
			if(__localThreadId < residue){
				t_s = __localThreadId*(rows_per_thread+1) + my_start_row + NLINES -1;
				t_e = t_s + rows_per_thread;
			}
			else{
				t_s = __localThreadId * rows_per_thread + residue + my_start_row + NLINES -1;
				t_e = t_s + rows_per_thread -1;
			}
			/*t_s = __localThreadId * rows_per_thread + my_start_row + NLINES -1;
			t_e = t_s + rows_per_thread -1;
			if(__localThreadId == __numLocalThreads-2)
				t_e = my_end_row -1;
				*/
		}

		float(*c_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])U_Curr;
		float(*n_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])U_Next;
		float *gather_local_convergence;
		if(__globalThreadId == 0)
			gather_local_convergence= new float[__numProcesses];
		/* computing utill convergence */
		int k =1;
		double jacbtime =0;
		while(1){
			if(k%100==0 && __globalThreadId ==0){
				std::cout<<"Iteration: "<<k<<std::endl;
			}
			switch(method){
			case 1:
				timer1.start();
				jacobi(U_Curr, U_Next, runtime, t_s, t_e, NLINES);
				jacbtime+=timer1.stop();
				break;
			case 2:
				gauss_seidel(U_Curr, U_Next);
				break;
			case 3:
				sor(U_Curr, U_Next);
				break;
			}
			//timer1.start();
			get_convergence_sqd(U_Curr, U_Next);
			//std::cout<<local_convergence_sqd<<std::endl;
			//runtime[3*__localThreadId +1] += timer1.stop();
			//timer2.start();
			SharedDataGather(&local_convergence_sqd, sizeof(float), gather_local_convergence, 0);
			if(__globalThreadId == 0){
				convergence_sqd = 0;
				for(int i=0; i< __numProcesses; i++)
					convergence_sqd += gather_local_convergence[i];
				convergence = sqrt(convergence_sqd);
				// std::cout<<"L2 = "<<convergence<<", "<<k<<std::endl;
			}
			SharedDataBcast(&convergence, sizeof(float), 0);
			if(convergence <= EPSILON){
				break;
			}
			/* copy U_Next to U_Curr */

			for (int j = t_start_row; j <= t_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					U_Curr[(j - my_start_row)*((int) floor (width / H)) +i] =
							U_Next[(j - my_start_row)*((int) floor (width / H)) +i];
				}
			}
			//runtime[3*__localThreadId +2] += timer2.stop();
			k++;
			inter_Barrier();
		}
		runtime[4*__localThreadId] = timer.stop();
		runtime[4*__localThreadId+3] = jacbtime;
		if(__globalThreadId ==0){
			delete gather_local_convergence;
			printf
				("Estimated time to convergence in %d iterations using %d processors on a %dx%d grid is %f seconds. %f\n",
				 k, __numProcesses, (int) floor (width / H), (int) floor (heigth / H),
				 timer.stop(), runtime[4*__localThreadId+1]);
		}
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl on "<<
					__processId<<".\n";
		}
	}

	~Heat2D(){
		if(U_Curr){
			free(U_Curr);
			free(U_Next);
			delete U_Curr_Above;
			delete U_Curr_Below;
			delete U_Above_Buffer;
			delete U_Below_Buffer;

			delete local_convergence_sqd_array;
		}
	}
	Heat2D()
	:t_start_row(this),
	 t_end_row(this),
	 t_num_rows(this){}

};

int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads=0;
	int nprocs=0;
	int width=20;
	int heigth=20;
	int method = 1;

	int opt;
		extern char* optarg;
		extern int optind;
		opt=getopt(argc, argv, "t:p:w:h:m:");
		while(opt!=EOF){
			switch(opt){
			case 't':
				nthreads = atoi(optarg);
				break;
			case 'p':
				nprocs = atoi(optarg);
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'h':
				heigth = atoi(optarg);
				break;
			case 'm':
				method = atoi(optarg);
				break;
			case '?':
				break;
			default:
				break;
			}
			opt=getopt(argc, argv, "t:p:w:h:m:");
		}

		int procs = ctx.numProcs();
		if(nprocs != procs){
			std::cerr<<"process number not match with arguments '-p' !!!\n";
			return 1;
		}
		int myproc = ctx.getProcRank();

		double *runtime = new double[4*nthreads];
		for(int i=0; i<4*nthreads; i++)
			runtime[i]=0;
		ProcList plist1;
		for(int i=0; i<nprocs; i++)
			for(int j=0; j<nthreads; j++)
				plist1.push_back(i);
		Task<Heat2D> heat2DInst(plist1);
		heat2DInst.init(width, heigth, method);
		heat2DInst.run(runtime);
		heat2DInst.wait();


		double avg_runtime1=0;
		double avg_runtime2=0;
		double avg_comptime1=0;
		double firsttime=0;
		double jacbtime =0;
		//double avg_commtime1 =0;
		for(int i=0; i<nthreads; i++){
			avg_runtime1 += runtime[4*i];
			jacbtime += runtime[4*i+3];
			//avg_comptime1 += runtime[3*i + 1];
			//avg_commtime1 += runtime[3*i + 2];
		}
		jacbtime /= nthreads;
		for(int i=0; i<nthreads-1; i++)
			firsttime+=runtime[4*i+2];
		firsttime /=(nthreads-1);
		avg_runtime1 /= nthreads;
		//avg_comptime1 /= nthreads;
		//avg_commtime1 /= nthreads;
		ctx.Barrier();
		MPI_Reduce(&avg_runtime1, &avg_runtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		avg_runtime2/=nprocs;
		if(myproc ==0){
			std::cout<<"average run() time: "<<avg_runtime2<<std::endl;
			std::cout<<"last thread run() time: "<<runtime[(nthreads-1)*4+2]<<std::endl;
			std::cout<<"first thread run() time: "<<firsttime<<std::endl;

			double timer[4];
			 timer[0]=avg_runtime1;
			 timer[1]=runtime[(nthreads-1)*4+2];
			 timer[2]=firsttime;
			 timer[3]=jacbtime;
			 print_time(4, timer);
		}
		ctx.Barrier();
		/*for(int i=0; i<nprocs; i++){
			if(i == myproc){
				std::cout<<"time info on proc "<<i<<":"<<std::endl;
				std::cout<<"\trun: "<<avg_runtime1<<
						" compute: "<<avg_comptime1<<
						" commu: "<<avg_runtime1-avg_comptime1<<std::endl;
				if(nthreads>1 && nprocs>1){
					for(int j=0; j<nthreads; j++)
						std::cout<<"\t\tthread "<<j<<": "<<runtime[3*j+2]<<std::endl;
				}
			}
			ctx.Barrier();
		}
		*/
		delete runtime;

		return 0;
}
