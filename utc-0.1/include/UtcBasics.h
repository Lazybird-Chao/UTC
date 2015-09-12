#ifndef UTC_BASICS_H_
#define UTC_BASICS_H_

#include <pthread.h>
#include <sys/types.h>


//#define UTC_DEBUG
//#define UTC_BAR_DEBUG
//#define PRINT_EXCEPTION
//#define UTC_DEVELOP
//#define XFER_DEBUG

//#define USE_RUN_FUNCTORS


#define USE_MPI_BASE
#define _STANDARD_MPI
//#define MULTIPLE_MPI

//#define PTHREAD_IS_STRUCT

#ifdef _STANDARD_MPI
	#include <mpi.h>
#endif


typedef int RankId;
typedef int ProcId;



#ifdef PTHREAD_IS_STRUCT
	typedef unsigned long TaskId;
	typedef unsigned long ThreadId;
#else
	typedef pthread_t TaskId;
	typedef pthread_t ThreadId;
#endif


const int MAX_PROCS=128;
const int MAX_TASKS=512;
const int MAX_THREADS=512;






#endif

