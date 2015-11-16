#ifndef UTC_BARRIER_H_
#define UTC_BARRIER_H_

#include <condition_variable>
#include <mutex>
#include <fstream>

#include "UtcBasics.h"
#ifdef USE_MPI_BASE
	#include <mpi.h>
#endif

namespace iUtc{

class Barrier{
public:
	Barrier();

	Barrier(int numLocalThreads, int taskid);

	Barrier(int numLocalThreads, int taskid, MPI_Comm *comm);

	~Barrier();

	// synch among threads of a task in one process
	void synch_intra(int local_rank);
	// synch among threads in a task, including all threads
	void synch_inter(int local_rank);


private:
	Barrier(const Barrier& other) = delete;
	Barrier& operator=(const Barrier& other) = delete;

	TaskId	m_taskId;

#ifdef USE_MPI_BASE
	MPI_Comm * m_taskCommPtr;
#endif

	//
	int m_numLocalThreads;
	std::mutex m_intraThreadSyncMutex;
	std::condition_variable m_intraThreadSyncCond;

	int m_intraThreadSyncCounterComing[2]; // two counters for rotate
	int m_intraThreadSyncCounterLeaving[2];
	// local thread i use m_countIdx[i] to specify which counter to use
	int *m_countIdx;

};


// utility functions
void intra_Barrier();

void inter_Barrier();


}// namespace iUtc


#endif

