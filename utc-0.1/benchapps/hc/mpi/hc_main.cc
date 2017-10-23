/*
 * hc_main.cc
 *
 * The sequential heat conduction program
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -v -h 100 -w 80 -e 0.001
 * 			-v: print time info
 * 			-h: 2D domain height
 * 			-w: 2D domain width
 * 			-e: convergence accuracy
 */


#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstdlib>
#include "mpi.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <vector>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#define FTYPE float
#define MPI_FTYPE MPI_FLOAT

#define H 1.0
#define T_SRC0 1550.0
#define ITERMAX 100		//

void init_domain(FTYPE *domain_ptr, int h, int w){
	for (int j = 0; j < (int)floor(h/H); j++) {
		for (int i = 0; i < (int) floor (w / H); i++) {
			domain_ptr[j*((int) floor (w / H)) + i] = 0.0;
		}
	}
}

 FTYPE get_convergence_sqd(FTYPE *current_ptr, FTYPE *next_ptr, int h, int w){
	FTYPE sum = 0.0;
	for(int i=0; i<(int)floor(h/H); i++){
		for(int j=0; j<(int) floor (w / H); j++){
			//sum += pow(current_ptr[i*((int) floor (w / H)) + j]-next_ptr[i*w+j],2);
			sum += (current_ptr[i*((int) floor (w / H)) + j]-next_ptr[i*w+j]) *
					(current_ptr[i*((int) floor (w / H)) + j]-next_ptr[i*w+j]);
		}
	}
	return sum;
}

inline void enforce_bc_par(FTYPE *domain_ptr, int i, int j, int h, int w,
		int startRowIndex,
		int total_rows){
	if(i==(w/2-1) && j+startRowIndex==0){
		domain_ptr[j*w + i] = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		domain_ptr[j*w + i] = 0.0;
	}
}

inline FTYPE get_var_par(
		FTYPE *domain_ptr,
		int i,
		int j,
		int h,
		int w,
		FTYPE *top_row,
		FTYPE *bottom_row,
		int startRowIndex,
		int total_rows){

	FTYPE ret_val;
	if(i == w/2-1 && j+startRowIndex==0){
		ret_val = T_SRC0;
	}
	else if(i<=0 || j+startRowIndex<=0 || i>=w-1 || j+startRowIndex>=total_rows-1){
		ret_val = 0.0;
	}
	else if(j<0 )
		ret_val = top_row[i];
	else if(j>h-1)
		ret_val = bottom_row[i];
	else
		ret_val = domain_ptr[j*w + i];

	return ret_val;
}

inline FTYPE f(int i, int j){
	return 0.0;
}

 void jacobi(FTYPE *current_ptr, FTYPE *next_ptr, int h, int w,
		FTYPE* top, FTYPE* bot, int start_rows, int totalrows){
	int i, j;
	for(j = 0; j<h; j++){
		for(i = 0; i<w; i++){
			next_ptr[j*w + i] =
					0.25 *
					(get_var_par(current_ptr, i-1, j, h, w, top, bot, start_rows, totalrows)+
							get_var_par(current_ptr, i+1, j, h, w, top, bot, start_rows, totalrows) +
							get_var_par(current_ptr, i, j-1, h, w, top, bot, start_rows, totalrows) +
							get_var_par(current_ptr, i, j+1, h, w, top, bot, start_rows, totalrows));

			enforce_bc_par(next_ptr, i, j, h, w, start_rows, totalrows);
		}
	}
}

int main(int argc, char**argv){
	int WIDTH = 400;
	int HEIGHT = 600;
	FTYPE EPSILON = 0.1;
	bool printTime = false;
	bool output = false;
	int nprocess = 1;

	/*
	 * run as ./a.out -v -h 100 -w 80 -e 0.001
	 * 		-v: print time info
	 * 		-h: 2D domain height
	 * 		-w: 2D domain width
	 * 		-e: convergence accuracy
	 */
	int opt;
	extern char* optarg;
	extern int optind, optopt;
	opt=getopt(argc, argv, "vh:w:e:op:");
	while(opt!=EOF){
		switch(opt){
		case 'v':
			printTime = true;
			break;
		case 'w':
			WIDTH = atoi(optarg);
			break;
		case 'h':
			HEIGHT = atoi(optarg);
			break;
		case 'e':
			EPSILON = atof(optarg);
			break;
		case 'o':
			output = true;
			break;
		case 'p':
			nprocess = atoi(optarg);
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
		opt=getopt(argc, argv, "vh:w:e:op:");
	}
	if(WIDTH<=0 || HEIGHT<=0){
		std::cerr<<"illegal width or height"<<std::endl;
		exit(1);
	}

	int procs;
	int myproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	if(nprocess != procs){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}

	int blockH = HEIGHT / nprocess;
	int start_rows = myproc * blockH;
	FTYPE *domain_matrix = nullptr;
	if(myproc == 0)
		domain_matrix = new FTYPE[HEIGHT*WIDTH];
	FTYPE *U_Curr = (FTYPE*)malloc(sizeof(FTYPE)*blockH*(int)floor(WIDTH/H));
	FTYPE *U_Next = (FTYPE*)malloc(sizeof(FTYPE)*blockH*(int)floor(WIDTH/H));
	init_domain(U_Curr, blockH, WIDTH);
	init_domain(U_Next, blockH, WIDTH);
	FTYPE *top_row = new FTYPE[WIDTH];
	FTYPE *bot_row = new FTYPE[WIDTH];
	init_domain(top_row, 1, WIDTH);
	init_domain(bot_row, 1, WIDTH);
	MPI_Barrier(MPI_COMM_WORLD);

	/*
	cpu_set_t cpuset;
	pthread_t thread;
	std::vector<int> ret;
	CPU_ZERO(&cpuset);
	thread = pthread_self();
	int s;
	s= pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
	if(s)
	{
		std::cerr<<"ERROR, Affinity get error!"<<std::endl;
	}
	usleep(100000*myproc);
	std::cout<<"rank: "<<myproc<<": ";
	for(int i=0; i<CPU_SETSIZE; i++){
		if(CPU_ISSET(i, &cpuset)){
			ret.push_back(i);
			std::cout<<i<<" ";
		}
	}
	std::cout<<std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	if(myproc == 0)
		std::cout<<"start computing...\n";

	/*
	 * main iterate computing
	 */
	double t1, t2;
	double totaltime = 0;
	double computetime = 0;
	double commtime = 0;
	MPI_Request req1, req2;
	t1 = MPI_Wtime();
	int iters = 1;
	//std::cout<<blockH<<" "<<start_rows<<std::endl;
	while(iters <= ITERMAX){
		if(iters % 1000 ==0 && myproc == 0){
			std::cout<<"iteration: "<<iters<<" ..."<<std::endl;
		}
		/* jacobi iterate */
		t2 = MPI_Wtime();
		jacobi(U_Curr, U_Next, blockH, WIDTH, top_row, bot_row, start_rows, HEIGHT);
		/*check if convergence */
		FTYPE convergence_sqd = get_convergence_sqd(U_Curr, U_Next, blockH, WIDTH);
		computetime += MPI_Wtime() - t2;

		t2 = MPI_Wtime();
		FTYPE total_converge;
		MPI_Reduce(&convergence_sqd, &total_converge, 1, MPI_FTYPE,
				MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(&total_converge, 1, MPI_FTYPE, 0, MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t2;
		//MPI_Barrier(MPI_COMM_WORLD);
		if(sqrt(total_converge) <= EPSILON)
			break;
		FTYPE *tmp = U_Curr;
		U_Curr = U_Next;
		U_Next = tmp;
		iters++;

		t2 = MPI_Wtime();
		if(myproc > 0){
			MPI_Send(U_Curr, WIDTH, MPI_FTYPE, myproc-1, 0, MPI_COMM_WORLD);
			//MPI_Isend(U_Curr, WIDTH, MPI_FTYPE, myproc-1, 0, MPI_COMM_WORLD, &req1);
		}
		if(myproc < nprocess -1){
			MPI_Recv(bot_row, WIDTH, MPI_FTYPE, myproc+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if(myproc < nprocess-1){
			MPI_Send(U_Curr + (blockH-1)*WIDTH, WIDTH, MPI_FTYPE, myproc+1, 1, MPI_COMM_WORLD);
			//MPI_Isend(U_Curr + (blockH-1)*WIDTH, WIDTH, MPI_FTYPE, myproc+1, 1, MPI_COMM_WORLD, &req2);
		}
		if(myproc > 0){
			MPI_Recv(top_row, WIDTH, MPI_FTYPE, myproc-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		commtime += MPI_Wtime() - t2;
	}
	t2 = MPI_Wtime();
	MPI_Gather(U_Next, blockH*WIDTH, MPI_FTYPE,
			   domain_matrix, blockH*WIDTH, MPI_FTYPE,
			   0, MPI_COMM_WORLD);
	commtime += MPI_Wtime() - t2;
	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	totaltime = t2 -t1;

	if(output && myproc == 0){
		char ofile[100] = "output.txt";
		FILE *fp = fopen(ofile, "w");
		for(int i=0; i<HEIGHT; i++){
			for(int j=0; j<WIDTH; j++){
				fprintf(fp, "%.5f ", domain_matrix[i*WIDTH +j]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
	}

	free(U_Curr);
	free(U_Next);
	delete top_row;
	delete bot_row;

	double runtime[3] = {0,0,0};
	MPI_Reduce(&totaltime, runtime+0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&computetime, runtime+1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&commtime, runtime+2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(myproc == 0){
		for(int i = 0; i< 3; i++)
			runtime[i] /= nprocess;
		std::cout<<"Test complete !!!"<<std::endl;
		if(printTime){
			std::cout<<"\tDomain size: "<<WIDTH<<" X "<<HEIGHT<<std::endl;
			std::cout<<"\tAccuracy: "<<EPSILON<<std::endl;
			std::cout<<"\tIterations: "<<iters<<std::endl;
			std::cout<<"\tTime info: \n";
			std::cout<<"\t\ttotaltime: "<<std::fixed<<std::setprecision(4)<<runtime[0]<<"(s)"<<std::endl;
			std::cout<<"\t\tcomputetime: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
			std::cout<<"\t\tcommtime: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		}

		for(int i = 0; i< 3; i++)
			runtime[i] *=1000;
		print_time(3, runtime);
	}

	MPI_Finalize();
	return 0;

}
