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
#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#define FTYPE float

#define H 1.0
#define T_SRC0 550.0
#define ITERMAX 100		// not used

void init_domain(FTYPE *domain_ptr, int h, int w){
	for (int j = 0; j < (int)floor(h/H); j++) {
		for (int i = 0; i < (int) floor (w / H); i++) {
			domain_ptr[j*((int) floor (w / H)) + i] = 0.0;
		}
	}
}

inline FTYPE get_convergence_sqd(FTYPE *current_ptr, FTYPE *next_ptr, int h, int w){
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

inline void enforce_bc_par(FTYPE *domain_ptr, int i, int j, int h, int w){
	if(i==((int)floor(w/H/2)-1) && j==0){
		domain_ptr[j*((int)floor(w/H)) + i] = T_SRC0;
	}
	else if(i<=0 || j<=0 || i>=(int)floor(w/H)-1 || j>=(int)floor(h/H)-1){
		domain_ptr[j*((int)floor(w/H)) + i] = 0.0;
	}
}

inline FTYPE get_var_par(FTYPE *domain_ptr, int i, int j, int h, int w){
	FTYPE ret_val;

	if(i == ((int)floor(w/H/2)-1) && j==0){
		ret_val = T_SRC0;
	}
	else if(i<=0 || j<=0 || i>=(int)floor(w/H)-1 || j>=(int)floor(h/H)-1){
		ret_val = 0.0;
	}
	else
		ret_val = domain_ptr[j*((int)floor(w/H)) + i];

	return ret_val;
}

inline FTYPE f(int i, int j){
	return 0.0;
}

inline void jacobi(FTYPE *current_ptr, FTYPE *next_ptr, int h, int w){
	int i, j;
	for(j = 0; j<(int)floor(h/H); j++){
		for(i = 0; i<(int) floor (w / H); i++){
			next_ptr[j*((int) floor (w / H)) + i] =
					0.25 *
					(get_var_par(current_ptr, i-1, j, h, w)+
							get_var_par(current_ptr, i+1, j, h, w) +
							get_var_par(current_ptr, i, j-1, h, w) +
							get_var_par(current_ptr, i, j+1, h, w));
							//(pow(H, 2)*f(i, j)));
			enforce_bc_par(next_ptr, i, j, h, w);
		}
	}
}

int main(int argc, char**argv){
	int WIDTH = 400;
	int HEIGHT = 600;
	FTYPE EPSILON = 0.1;
	bool printTime = false;
	bool output = false;

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
	opt=getopt(argc, argv, "vh:w:e:o");
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
		case ':':
			std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
			break;
		case '?':
			std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "vh:w:e:o");
	}
	if(WIDTH<=0 || HEIGHT<=0){
		std::cerr<<"illegal width or height"<<std::endl;
		exit(1);
	}

	FTYPE *U_Curr = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	FTYPE *U_Next = (FTYPE*)malloc(sizeof(FTYPE)*(int)floor(HEIGHT/H)*(int)floor(WIDTH/H));
	init_domain(U_Curr, HEIGHT, WIDTH);
	init_domain(U_Next, HEIGHT, WIDTH);

	/*
	 * main iterate computing
	 */
	std::cout<<"start computing...\n";
	double t1, t2;
	t1 = getTime();
	int iters = 1;
	while(1){
		if(iters % 1000 ==0)
			std::cout<<"iteration: "<<iters<<" ..."<<std::endl;
		/* jacobi iterate */
		jacobi(U_Curr, U_Next, HEIGHT, WIDTH);
		/*check if convergence */
		FTYPE convergence_sqd = get_convergence_sqd(U_Curr, U_Next, HEIGHT, WIDTH);
		if(sqrt(convergence_sqd) <= EPSILON)
			break;
		FTYPE *tmp = U_Curr;
		U_Curr = U_Next;
		U_Next = tmp;
		iters++;
	}
	t2 = getTime();
	double runtime = t2 -t1;

	if(output){
		char ofile[100] = "output.txt";
		FILE *fp = fopen(ofile, "w");
		for(int i=0; i<HEIGHT; i++){
			for(int j=0; j<WIDTH; j++){
				fprintf(fp, "%.5f ", U_Next[i*WIDTH +j]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
	}

	free(U_Curr);
	free(U_Next);

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"\tDomain size: "<<WIDTH<<" X "<<HEIGHT<<std::endl;
		std::cout<<"\tAccuracy: "<<EPSILON<<std::endl;
		std::cout<<"\tIterations: "<<iters<<std::endl;
		std::cout<<"\tTime info: "<<std::fixed<<std::setprecision(4)<<runtime<<"(s)"<<std::endl;
	}

	print_time(1, &runtime);

	return 0;

}
