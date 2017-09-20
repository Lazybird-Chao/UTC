/*
 * test_gsData_internal_shmem.cc
 *
 *  Created on: Sep 14, 2017
 *      Author: chaoliu
 */

#include "Utc.h"
#include "../../../benchapps/common/helper_getopt.h"
#include "../../../benchapps/common/helper_err.h"
#include "../../../benchapps/common/helper_printtime.h"

#include <iostream>
#include <string>

using namespace iUtc;



int main(int argc, char* argv[]){
	UtcContext &ctx = UtcContext::getContext(argc, argv);

	int nthreads;
	int nprocs;

	int opt;
	extern char* optarg;
	extern int optind;
	opt=getopt(argc, argv, "t:p:");
	while(opt!=EOF){
		switch(opt){
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'p':
			nprocs = atoi(optarg);
			break;
		case '?':
			break;
		default:
			break;
		}
		opt=getopt(argc, argv, "t:p:");
	}

	if(nprocs != ctx.numProcs()){
		std::cerr<<"process number not match with arguments '-p' !!!\n";
		return 1;
	}

}

