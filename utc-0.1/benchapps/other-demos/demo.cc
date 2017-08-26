/*
 * demo.cc
 *
 */

#include "usertask.h"
#include "../common/helper_getopt.h"
#include "../common/helper_err.h"
#include "Utc.h"
using namespace iUtc;

int main(int argc, char **argv){
	long vecSize = 1000;
	// initialize the utc environment
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "s:");
	while(opt!=EOF){
		switch(opt){
		case 's':
			vecSize = atol(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "s:");
	}

	/* define and create task object
	 * ProcList indicate which process the task is going to run on:
	 * like:[0,0] means run two threads on process0
	 * 		[0,0,1] means two threads on process0 and 1 threads on process1
	 * 	when running program with multiple process, need use mpirun to start
	 * 	running
	 * */
	Task<vectorGen<double>> genT(ProcList(0));
	Task<vectorOp<double>> addT(ProcList(0));
	Task<vectorOp<double>> subT(ProcList(0));
	Task<vectorOp<double>> mulT(ProcList(0));
	Task<vectorOp<double>> divT(ProcList(0));

	/*
	 * create conduit, construct a conduit with two task obj.
	 */
	std::vector<Conduit *> cdt1;
	cdt1.push_back(new Conduit(&genT, &addT));
	cdt1.push_back(new Conduit(&genT, &subT));
	cdt1.push_back(new Conduit(&genT, &mulT));
	cdt1.push_back(new Conduit(&genT, &divT));
	Task<vectorMul<double>> vmulT1(ProcList(0));
	Task<vectorMul<double>> vmulT2(ProcList(0));
	Conduit* cdt2[4];
	cdt2[0] = new Conduit(&addT, &vmulT1);
	cdt2[1] = new Conduit(&mulT, &vmulT1);
	cdt2[2] = new Conduit(&subT, &vmulT2);
	cdt2[3] = new Conduit(&divT, &vmulT2);


	/*
	 * invoke initImpl method of user-task template.
	 *
	 */
	genT.init(vecSize, cdt1);
	addT.init(vecSize, cdt1[0], cdt2[0]);
	mulT.init(vecSize, cdt1[2], cdt2[1]);
	subT.init(vecSize, cdt1[1], cdt2[2]);
	divT.init(vecSize, cdt1[3], cdt2[3]);
	vmulT1.init(vecSize, cdt2[0], cdt2[1]);
	vmulT2.init(vecSize, cdt2[2], cdt2[3]);


	double result[10][2];
	double t[7], t1=0;
	Timer timer;
	timer.start();
	for(int i=0; i<10; i++){
		/*
		 * invoke runImpl method of user-task template.
		 *
		 */
		genT.run(i, &t[0]);

		addT.run(Op::add, &t[1]);
		subT.run(Op::sub, &t[2]);
		mulT.run(Op::mul, &t[3]);
		divT.run(Op::div, &t[4]);

		vmulT1.run(&result[i][0], &t[5]);
		vmulT2.run(&result[i][1], &t[6]);

		/*
		 * call wait method to wait for the last run-call finish
		 * as run-call is unblocking.
		 *
		 * Becasue there are dependency between tasks, so we just call
		 * wait on these two, no need to call wait for other.
		 *
		 * wait-call is blocking, the main program will wait here until
		 * the task's work complete
		 */
		vmulT1.wait();
		vmulT2.wait();
		for(int j=0; j<7; j++)
			t1+= t[j];
	}
	double t2 = timer.stop();

	/*
	 * finish-call to terminate the threads of the task. onece called
	 * the task related thread is finished and can not use this task to run.
	 * Otherwise, the task related thread is suspending in background.
	 *
	 */
	genT.finish();


	std::cout<<"sum of compute time: "<<t1<<std::endl;
	std::cout<<"total time: "<<t2<<std::endl;

	return 0;
}



