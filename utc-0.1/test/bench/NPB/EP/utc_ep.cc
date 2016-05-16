/*
 *
 */

#include "../../helper_getopt.h"

#include "type.h"
#include "npbparams.h"
#include "randdp.h"
//#include "timers.h"
#include "print_results.h"

#include "Utc.h"
#include "../../helper_printtime.h"
#include <iostream>
#include <string>

using namespace iUtc;

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

#define MK        16
#define MM        (_M - MK)
#define NN        (1 << MM)
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0


class ep_kernel_compute{
public:
	void init( double *sx, double *sy, double *gc, double* q, double* timeVal){
		myTrank = getTrank();
		myLrank = getLrank();
		if(getUniqueExecution()){
			nTotalthreads = getGsize();
			nLocalthreads = getLsize();
			this->sx_out = sx;
			this->sy_out = sy;
			this->q_out = q;
			this->gc_out = gc;
			this->timeVal = timeVal;
			sbarrier.set(nLocalthreads);
			this->sx =0.0;
			this->sy =0.0;
			for(int i=0; i<NQ;i++)
				this->q[i]=0.0;
		}
		intra_Barrier();
		int np_add;
		np = NN / nTotalthreads;
		int no_large_nodes = NN % nTotalthreads;
		if(myTrank < no_large_nodes)
			np_add = 1;
		else
			np_add =0;
		this->np = np + np_add;
		if(this->np==0){
			std::cerr<<"Too many threads: "<<nTotalthreads<<" "<<NN<<std::endl;
			exit(1);
		}
		if(np_add == 1)
			k_offset= this->np * myTrank -1;
		else
			k_offset= no_large_nodes*(this->np+1) + (this->np)*(myTrank - no_large_nodes)-1;

		/*if(myLrank ==0)
			std::cout<<"Finish init()"<<np<<" "<<nTotalthreads<<std::endl;*/
	}

	void run(){
		double dum[3]={1.0, 1.0,1.0};
		double x[2*NK];
		double t1, t2, t3, t4, x1, x2;
		double an, tt;
		double local_sx, local_sy;
		double local_q[NQ];
		double *sx_forgather, *sy_forgather, *q_forgather;
		if(myTrank==0){
			sx_forgather = new double[getPsize()];
			sy_forgather = new double[getPsize()];
			q_forgather = new double[getPsize()*NQ];
		}

		vranlc(0, &dum[0], dum[1], &dum[2]);
		dum[0] = randlc(&dum[1], dum[2]);
		for( int i=0; i<2*NK; i++)
			x[i]= -1.0e99;
		double local_timeVal[4];
		Timer timer1, timer2;
		for(int i=0; i<4; i++)
			local_timeVal[i] = 0.0;
		inter_Barrier();
		timer1.start();

		t1 = A;
		vranlc(0, &t1, A, x);

		//--------------------------------------------------------------------
		//  Compute AN = A ^ (2 * NK) (mod 2^46).
		//--------------------------------------------------------------------
		t1 = A;

		for(int i=0; i<MK +1; i++)
			t2 = randlc(&t1, t1);
		//std::cout<<t1<<" "<<t2<<std::endl;
		an = t1;
		tt = S;
		local_sx = 0.0;
		local_sy = 0.0;
		for(int i=0; i< NQ; i++)
			local_q[i]=0.0;

		for(int k=1; k<= np; k++){
			int kk = k_offset + k;
			t1 = S;
			t2 = an;

			// Find starting seed t1 for this kk.
			for( int i=1; i<=100; i++){
				int ik= kk/2;
				if((2*ik) != kk) t3 = randlc(&t1, t2);
				if(ik == 0) break;
				t3 = randlc(&t2, t2);
				kk = ik;
			}

			//--------------------------------------------------------------------
			//  Compute uniform pseudorandom numbers.
			//--------------------------------------------------------------------
			timer2.start();
			vranlc(2*NK, &t1, A, x);
			local_timeVal[2] += timer2.stop(); // t_randn

			//--------------------------------------------------------------------
			//  Compute Gaussian deviates by acceptance-rejection method and
			//  tally counts in concentri//square annuli.  This loop is not
			//  vectorizable.
			//--------------------------------------------------------------------
			timer2.start();
			for (int i = 0; i < NK; i++) {
				x1 = 2.0 * x[2*i] - 1.0;
				x2 = 2.0 * x[2*i+1] - 1.0;
				t1 = x1 * x1 + x2 * x2;
				if (t1 <= 1.0) {
					t2    = sqrt(-2.0 * log(t1) / t1);
					t3    = (x1 * t2);
					t4    = (x2 * t2);
					int l     = MAX(fabs(t3), fabs(t4));
					local_q[l] = local_q[l] + 1.0;
					local_sx    = local_sx + t3;
					local_sy    = local_sy + t4;
				}
			}
			local_timeVal[1] += timer2.stop();  // t_gpairs

		}

		timer2.start();
		rwlock.lock();
		for(int i=0; i<NQ; i++){
			q[i] +=local_q[i];
		}
		sx += local_sx;
		sy += local_sy;
		rwlock.unlock();
		sbarrier.wait();

		SharedDataGather(&sx, sizeof(double), sx_forgather, 0);
		SharedDataGather(&sy, sizeof(double), sy_forgather, 0);
		SharedDataGather(q, sizeof(double)*NQ, q_forgather, 0);
		local_timeVal[3] = timer2.stop();   // t_rcomm

		if(myTrank ==0){
			for(int ii=1; ii<getPsize(); ii++){
				sx += sx_forgather[ii];
				sy += sy_forgather[ii];
				for(int jj=0; jj< NQ; jj++){
					q[jj] += q_forgather[ii*NQ+jj];
				}
			}
			*gc_out =0;
			for(int i=0; i<NQ; i++)
				*gc_out += q[i];
		}
		//SharedDataBcast(&sx, sizeof(double), 0);
		//SharedDataBcast(&sy, sizeof(double), 0);
		//SharedDataBcast(q, sizeof(double)*NQ, 0);
		sbarrier.wait();
		local_timeVal[0] = timer1.stop();

		rwlock.lock();
		for(int i=0; i<4; i++)
			timeVal[i] += local_timeVal[i];
		rwlock.unlock();
		sbarrier.wait();
		if(myTrank ==0){
			*sx_out = sx;
			*sy_out = sy;
			for(int i=0; i<NQ; i++)
				q_out[i]=q[i];
		}
		if(myLrank==0){
			for(int i=0; i<4; i++)
				timeVal[i] = timeVal[i]/nLocalthreads;
		}


	}

private:
	double sx, sy, *timeVal;
	double q[NQ];
	double *sx_out, *sy_out, *q_out, *gc_out;
	static thread_local int np;
	static thread_local int k_offset;
	int nTotalthreads;
	int nLocalthreads;
	static thread_local int myTrank;
	static thread_local int myLrank;

	SpinBarrier sbarrier;
	SpinLock rwlock;
};
thread_local int ep_kernel_compute::np;
thread_local int ep_kernel_compute::k_offset;
thread_local int ep_kernel_compute::myTrank;
thread_local int ep_kernel_compute::myLrank;


void usage() {
    std::string help =
        "use option: -t nthreads\n   :threads per node of Task\n"
		"            -p nprocs       :number of nodes running on \n";
    std::cerr<<help.c_str();
    exit(-1);
}

int main(int argc, char* argv[]){

	UtcContext ctx(argc, argv);
	int nthreads, nprocs;
	int opt;
	extern char *optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:");
	while( opt!=EOF ){
		switch (opt){
			case 't':
				nthreads = atoi(optarg);
				break;
			case 'p':
				nprocs = atoi(optarg);
				break;
			case '?':
				usage();
				break;
			default:
				usage();
				break;
		}
		opt=getopt(argc, argv, "t:p:");
	}

	int np;
	char size[16];
	double Mops;
	double sx_verify_value, sy_verify_value, sx_err, sy_err;
	double sx, sy, gc;
	int nit;
	bool verified;
	double q[NQ];

	if(ctx.getProcRank()==0){
		sprintf(size, "%15.0lf", pow(2.0, _M+1));
		int j = 14;
		if (size[j] == '.') j--;
		size[j+1] = '\0';
		printf("\n\n NAS Parallel Benchmarks (NPB3.3-SER-C) - EP Benchmark\n");
		printf("\n Number of random numbers generated: %15s\n", size);
	}

	/*if(ctx.getProcRank()==0)
		std::cout<<nprocs<<" "<<nthreads<<std::endl;*/
	std::vector<int> rvec;
	for(int i=0; i< nprocs; i++)
			for(int j=0; j<nthreads;j++)
				rvec.push_back(i);
	ProcList rlist(rvec);
	Task<ep_kernel_compute> ep_instance("ep-compute", rlist);

	double timeVal[4]={0,0,0,0};
	ep_instance.init(&sx, &sy, &gc, q, timeVal);

	ep_instance.run();

	ep_instance.wait();

	ep_instance.finish();


	double timeValgather[4];
	MPI_Reduce(timeVal, timeValgather, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(ctx.getProcRank()==0){
		nit = 0;
		verified = true;
		if (_M == 24) {
			sx_verify_value = -3.247834652034740e+3;
			sy_verify_value = -6.958407078382297e+3;
		} else if (_M == 25) {
			sx_verify_value = -2.863319731645753e+3;
			sy_verify_value = -6.320053679109499e+3;
		} else if (_M == 28) {
			sx_verify_value = -4.295875165629892e+3;
			sy_verify_value = -1.580732573678431e+4;
		} else if (_M == 30) {
			sx_verify_value =  4.033815542441498e+4;
			sy_verify_value = -2.660669192809235e+4;
		} else if (_M == 32) {
			sx_verify_value =  4.764367927995374e+4;
			sy_verify_value = -8.084072988043731e+4;
		} else if (_M == 36) {
			sx_verify_value =  1.982481200946593e+5;
			sy_verify_value = -1.020596636361769e+5;
		} else if (_M == 40) {
			sx_verify_value = -5.319717441530e+05;
			sy_verify_value = -3.688834557731e+05;
		} else {
			verified = false;
		}

		if (verified) {
			sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
			sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
			verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
		}

		for(int i=0; i<4; i++)
			timeVal[i] = timeValgather[i]/ctx.numProcs();

		Mops = pow(2.0, _M+1) / timeVal[0] / 1000000.0;

		printf("\nEP Benchmark Results:\n\n");
		printf("CPU Time =%10.4lf\n", timeVal[0]);
		printf("N = 2^%5d\n", _M);
		printf("No. Gaussian Pairs = %15.0lf\n", gc);
		printf("Sums = %25.15lE %25.15lE\n", sx, sy);
		printf("Counts: \n");
		for (int i = 0; i < NQ; i++) {
			printf("%3d%15.0lf\n", i, q[i]);
		}

		print_results("EP", CLASS, _M+1, 0, 0, nit,
			  timeVal[0], Mops,
			  "Random numbers generated",
			  verified, NPBVERSION, COMPILETIME, CS1,
			  CS2, CS3, CS4, CS5, CS6, CS7);

		double tm = timeVal[0];
		if(tm <=0.0) tm=1.0;
		double tt = timeVal[0];
		printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
		tt = timeVal[1];
		printf("Gaussian pairs: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
		tt = timeVal[2];
		printf("Random numbers: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
		tt = timeVal[3];
		printf("Communications: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);


		// record timeinfo to file
		print_time(4, timeVal);

	}

	// print timers


	return 0;
}
