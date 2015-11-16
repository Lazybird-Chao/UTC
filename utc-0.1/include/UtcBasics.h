#ifndef UTC_BASICS_H_
#define UTC_BASICS_H_


/*
 *
 */
#define USE_DEBUG_LOG
#define USE_DEBUG_ASSERT




/*
 *
 */
#ifdef USE_DEBUG_LOG
#include <chrono>
#include <ctime>
extern std::chrono::system_clock::time_point SYSTEM_START_TIME;
#define PRINT_TIME_NOW(outstream) 	{ std::chrono::system_clock::time_point t_now=\
											std::chrono::system_clock::now();\
									time_t tt = std::chrono::system_clock::to_time_t(t_now);\
									char time_str[100]; \
									std::strftime(time_str, sizeof(time_str), "%F %T", \
												std::localtime(&tt));\
									std::chrono::system_clock::duration dtn =\
												t_now-SYSTEM_START_TIME;\
									outstream<<"[SYSTEM LOG]"<<time_str<<\
											"("<<dtn.count()<<")"<<">>>>>>>>>>:    ";}
#endif


/*
 *
 */
#define USE_CPLUS_THREAD_CREATION
#if defined(USE_CPLUS_THREAD_CREATION)
    #include <thread>
#elif defined(USE_BOOST_THREAD_CREATION)
    #include <boost/thread/thread.hpp>
#elif defined(USE_POSIX_THREAD_CREATION)
    #include <pthread.h>
    #include <sys/types.h>
#endif


/*
 *
 */
#define USE_MPI_BASE
#ifdef USE_MPI_BASE
    #include <mpi.h>
#endif
#define MULTIPLE_THREAD_MPI   //  MPI_THREAD_SINGLE,
                              //  MPI_THREAD_FUNNELED,
                              //  MPI_THREAD_SERIALIZED,

/*
 *
 */
typedef  int RankId;  // >=0
typedef  int ProcRank; // >=0
typedef  int ThreadRank; // >=0
typedef  int TaskId; // >=0
typedef  int ConduitId; // >=0
typedef  int MessageTag; // >=0

#if defined(USE_CPLUS_THREAD_CREATION)
    typedef  std::thread::id ThreadId;
#elif defined(USE_BOOST_THREAD_CREATION)
    typedef boost::thread::id TreadId
#elif defined(USE_POSIX_THREAD_CREATION)
    typedef  pthread_t ThreadId;
#endif


/*
 *
 */
const int MAX_PROCS=32;
const int LOG_MAX_PROCS=5;
const int MAX_TASKS=128;
const int LOG_MAX_TASKS=7;
const int MAX_THREADS=256;
const int LOG_MAX_THREADS=8;

const int MAX_CONDUITS = 1024;
const int LOG_MAX_CONDUITS = 10;
const int INPROC_CONDUIT_CAPACITY_DEFAULT = 8;
const int INPROC_CONDUIT_CAPACITY_MAX = 32;
const int SMALL_MESSAGE_CUTOFF = (1024*1024);
const int NO_FINISHED_OP_MAX = 16;
const int CONDUIT_BUFFER_SIZE = 4096; // 4k bytes



#endif

