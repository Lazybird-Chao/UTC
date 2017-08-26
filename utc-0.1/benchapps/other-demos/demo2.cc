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

	Task<vectorGen<double>> genT(ProcList(0));
	Task<vectorOp<double>> addT(ProcList(0));
	Task<vectorOp<double>> subT(ProcList(0));
	Task<vectorOp<double>> mulT(ProcList(0));
	Task<vectorOp<double>> divT(ProcList(0));
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


	genT.init(vecSize, cdt1);
	addT.init(vecSize, cdt1[0], cdt2[0]);
	mulT.init(vecSize, cdt1[2], cdt2[1]);
	subT.init(vecSize, cdt1[1], cdt2[2]);
	divT.init(vecSize, cdt1[3], cdt2[3]);
	vmulT1.init(vecSize, cdt2[0], cdt2[1]);
	vmulT2.init(vecSize, cdt2[2], cdt2[3]);

	double result[10][2];
	Timer timer;
	timer.start();
	for(int i=0; i<1; i++){
		genT.run(i);

		addT.run(Op::add);
		subT.run(Op::sub);
		mulT.run(Op::mul);
		divT.run(Op::div);

		vmulT1.run(&result[i][0]);
		vmulT2.run(&result[i][1]);
		vmulT1.wait();
		vmulT2.wait();
	}
	double t1 = timer.stop();

	double r2[10][2];
	timer.start();
	compare(vecSize, r2);
	double t2 = timer.stop();


	std::cout<<"runtime: "<<t1<<"  "<<t2<<std::endl;

	return 0;
}



