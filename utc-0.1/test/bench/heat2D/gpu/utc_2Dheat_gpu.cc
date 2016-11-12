/*
 * utc_2Dheat_gpu.cc
 *
 *  Created on: Nov 8, 2016
 *      Author: chao
 */

#include "Utc.h"
#include "UtcGpu.h"
#include "../../helper_getopt.h"
#include "../../helper_printtime.h"
#include "utc_2Dheat_gpu_kernel.h"

#include <iostream>

#define H 1.0
#define _EPSILON 0.1

#define T_SRC0 550.0

using namespace iUtc;

class Heat2DGPU: public UserTaskBase{
private:
	int width;
	int heigth;
	int method;
	float EPSILON = _EPSILON;
	float convergence;

	float *U_Curr;
	float *U_Next;
	//float *U_Curr_Above;
	//float *U_Curr_Below;

	int my_start_row;
	int my_end_row;
	int my_num_rows;

	float *U_Curr_d;
	float *U_Next_d;
	//float *U_Curr_Above_d;
	//float *U_Curr_Below_d;

	int blocksize[2];

	inline int get_start(int rank){
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

	inline int get_end(int rank ){
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

	inline int get_num_rows(int rank){
		return 1+ get_end(rank) - get_start(rank);
	}

	 inline int global_to_local(int rank, int row){
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

	float get_convergence_sqd(float *current_ptr, float *next_ptr){
		float sum = 0;
		float(*c_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])current_ptr;
		float(*n_p)[(int) floor (width / H)] = (float(*)[(int) floor (width / H)])next_ptr;
		for(int j = my_start_row; j<= my_end_row; j++){
			for(int i = 0; i<(int) floor (width / H); i++){
				/*sum +=(n_p[j-my_start_row][i] -
						 c_p[j-my_start_row][i]) * (n_p[j-my_start_row][i] -
								 c_p[j-my_start_row][i]);*/
				sum +=
					pow (n_p[(j-my_start_row)][i] -
							c_p[(j-my_start_row)][i], 2);
			}
		}
		return sum;
	}


public:
	void initImpl(int w, int h, int meth, int blocksize[]){
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

			//U_Curr_Above = (float*)malloc(sizeof(float)*(int) floor (width / H));
			//U_Curr_Below = (float*)malloc(sizeof(float)*(int) floor (width / H));

			this->blocksize[0] = blocksize[0];
			this->blocksize[1] = blocksize[1];

		}
		inter_Barrier();
		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish initImpl on "
					<<__processId<<".\n";
		}
	}

	void runImpl(double *runtime){
		std::cout<<"begin run ..."<<std::endl;
		Timer t1, t2;

		t1.start();
		checkCudaRuntimeErrors(
				cudaMalloc(&U_Curr_d, sizeof (float) * my_num_rows *
									(int) floor (width / H)));
		checkCudaRuntimeErrors(
				cudaMalloc(&U_Next_d, sizeof (float) * my_num_rows *
									(int) floor (width / H)));
		runtime[0] = t1.stop();

		t1.start();
		t2.start();
		checkCudaRuntimeErrors(
				cudaMemcpy(U_Curr_d, U_Curr, sizeof (float) * my_num_rows *
									(int) floor (width / H), cudaMemcpyHostToDevice));
		checkCudaRuntimeErrors(
				cudaMemcpy(U_Next_d, U_Next, sizeof (float) * my_num_rows *
									(int) floor (width / H), cudaMemcpyHostToDevice));
		runtime[1] += t1.stop();

		float *pf[2];
		pf[0] = U_Curr_d;
		pf[1] = U_Next_d;
		int curf = 0;
		GpuKernel mykernel;
		mykernel.setGridDim((width + blocksize[0] -1)/blocksize[0],
				(my_num_rows + blocksize[1] -1)/blocksize[1]);
		mykernel.setBlockDim(blocksize[0], blocksize[1]);
		mykernel.setNumArgs(6);
		//mykernel.setArgs<float*>(0, pf[curf]);
		//mykernel.setArgs<float*>(1, pf[1-curf]);
		mykernel.setArgs<int>(2, my_start_row);
		mykernel.setArgs<int>(3, my_end_row);
		mykernel.setArgs<int>(4, width);
		mykernel.setArgs<int>(5, heigth);

		int k=1;
		while(1){
			if(k%100 ==0 && __globalThreadId ==0){
				std::cout<<"Iteration: "<<k<<std::endl;
			}

			t1.start();
			switch(method){
			case 1:
				mykernel.setArgs<float*>(0, pf[curf]);
				mykernel.setArgs<float*>(1, pf[1-curf]);
				mykernel.launchKernel((const void*)&jacobi_kernel);
				break;
			case 2:
				break;
			case 3:
				break;
			}
			runtime[2] += t1.stop();

			t1.start();
			checkCudaRuntimeErrors(
				cudaMemcpy(U_Next, pf[1-curf], sizeof (float) * my_num_rows *
									(int) floor (width / H), cudaMemcpyDeviceToHost));
			runtime[3]+= t1.stop();

			t1.start();
			float convergence_sqd = get_convergence_sqd(U_Curr, U_Next);
			runtime[4] += t1.stop();
			float convergence = sqrt(convergence_sqd);
			if(convergence <= EPSILON){
				break;
			}
			for (int j = my_start_row; j <= my_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					U_Curr[(j - my_start_row)*((int) floor (width / H)) +i] =
							U_Next[(j - my_start_row)*((int) floor (width / H)) +i];
				}
			}
			k++;
			curf = 	1-curf;

		}
		runtime[5] = t2.stop();
		runtime[6] = runtime[1]+runtime[2]+runtime[3];

		if(width < 200 && heigth < 200){
			long err = compare();
			if(err > 0){
				std::cout<<"err ! "<<err<<std::endl;
			}
			else{
				std::cout<<"correct !"<<std::endl;
			}
		}

		if(__localThreadId ==0){
			std::cout<<"task: "<<getCurrentTask()->getName()<<" finish runImpl on "<<
					__processId<<".\n";
		}
	}

	long compare(float eps = 1e-6){
		float *U_c = (float *) malloc (sizeof (float) * my_num_rows *
								(int) floor (width / H));
		float *U_n = (float *) malloc (sizeof (float) * my_num_rows *
								(int) floor (width / H));
		init_domain(U_c);
		init_domain(U_n);
		while(1){
			for (int j = my_start_row; j <= my_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					U_n[(j - my_start_row)*width + i] =
										.25 * (
										get_val_par( U_c, i - 1,j, width, heigth, my_start_row)
									+ get_val_par ( U_c, i + 1, j,width,heigth, my_start_row)
									+ get_val_par ( U_c, i, j - 1,width, heigth,my_start_row)
									+ get_val_par ( U_c,i, j + 1,width, heigth,my_start_row)
									- (pow (H, 2) * f (i, j))
									);
					enforce_bc_par (U_n, i, j, width, my_start_row);
				}
			}
			float convergence_sqd = get_convergence_sqd(U_c, U_n);
			float convergence = sqrt(convergence_sqd);
			if(convergence <= EPSILON){
				break;
			}
			for (int j = my_start_row; j <= my_end_row; j++) {
				for (int i = 0; i < (int) floor (width / H); i++) {
					U_c[(j - my_start_row)*((int) floor (width / H)) +i] =
							U_n[(j - my_start_row)*((int) floor (width / H)) +i];
				}
			}
		}
		long err = 0;
		for (int j = my_start_row; j <= my_end_row; j++) {
			for (int i = 0; i < (int) floor (width / H); i++){
				float tmp = fabs(U_n[(j - my_start_row)*((int) floor (width / H)) +i] -
						U_Next[(j - my_start_row)*((int) floor (width / H)) +i]);
				if( tmp > eps)
				{
					std::cout<<tmp<<std::endl;
					err++;
				}
			}
		}
		return err;

	}

	~Heat2DGPU(){
		if(U_Curr){
			free(U_Curr);
			free(U_Next);
		}

		if(U_Curr_d){
			cudaFree(U_Curr_d);
			cudaFree(U_Next_d);
		}
	}

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
		if(nthreads != 1){
			std::cerr<<"only run with 1 thread for this program\n";
			return 1;
		}

		double runtime[7]={0,0,0,0,0,0,0};
		int blocksize[2] = {16,16};
		Task<Heat2DGPU> myHeat2D(ProcList(0), TaskType::gpu_task);
		myHeat2D.init(width, heigth, method, blocksize);

		myHeat2D.run(runtime);

		myHeat2D.wait();
		myHeat2D.finish();

		if(myproc==0){
			std::cout<<"total time: "<<runtime[5]<<std::endl;
			std::cout<<"mem alloc time: "<<runtime[0]<<std::endl;
			std::cout<<"mem copyin time: "<<runtime[1]<<std::endl;
			std::cout<<"kernel run time: "<<runtime[2]<<std::endl;
			std::cout<<"mem copyout time: "<<runtime[3]<<std::endl;
			std::cout<<"conver compute time: "<<runtime[4]<<std::endl;

			print_time(7, runtime);
		}


		return 0;

}


