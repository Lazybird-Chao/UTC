/*
 * SpinBarrier.h
 *
 *  Created on: Nov 28, 2016
 *      Author: chao
 */

#ifndef UTC_SPINBARRIER_H_
#define UTC_SPINBARRIER_H_


#include <atomic>

namespace iUtc{

class SpinBarrier{
public:
	SpinBarrier();

	SpinBarrier(int nthreads);

	void set(int nthreads);

	void wait(int id=0);

private:
	std::atomic<int> m_barrierCounter;
	std::atomic<int> m_generation;
	int m_numThreadsForSync;
	std::atomic<int> m_barrierReady;
};

void intra_SpinBarrier();


}



#endif /* SPINBARRIER_H_ */
