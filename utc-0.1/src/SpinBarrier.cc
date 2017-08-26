/*
 * SpinBarrier.cc
 *
 *  Created on: Nov 28, 2016
 *      Author: chao
 */

#include "SpinBarrier.h"
#include "UtcBasics.h"
#include "TaskManager.h"
#include "TimerUtilities.h"

namespace iUtc{

SpinBarrier::SpinBarrier(){
	m_numThreadsForSync =0;
	m_barrierCounter=0;
	m_generation=0;
	m_barrierReady=0;
}

SpinBarrier::SpinBarrier(int nthreads){
	m_numThreadsForSync = nthreads;
	m_barrierCounter=0;
	m_generation=0;
	m_barrierReady=nthreads;
}

void SpinBarrier::set(int nthreads){
	if(m_numThreadsForSync!=nthreads){
		m_barrierReady.store(nthreads);
		m_numThreadsForSync = m_barrierReady;
	}
}

void SpinBarrier::wait(int id){
	// wait set() finish
	long _counter=0;
	while(m_barrierReady.load() != m_numThreadsForSync){
		_counter++;
		/*if(_counter<USE_PAUSE)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP){
			__asm__ __volatile__ ("pause" ::: "memory");
			std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);
			*/
		spinWait(_counter, id);

	}

	int generation = m_generation.load();
	int threadsForSync = m_numThreadsForSync;
	//
	m_barrierCounter.fetch_add(1);
	if(m_barrierCounter.compare_exchange_strong(threadsForSync, 0, std::memory_order_release)){
		m_generation.fetch_add(1);
		return;
	}
	_counter=0;
	while(m_generation.load()==generation){
		_counter++;
		/*if(_counter<USE_PAUSE)
			_mm_pause();
		else if(_counter<USE_SHORT_SLEEP){
			__asm__ __volatile__ ("pause" ::: "memory");
			std::this_thread::yield();
		}
		else if(_counter<USE_LONG_SLEEP)
			nanosleep(&SHORT_PERIOD, nullptr);
		else
			nanosleep(&LONG_PERIOD, nullptr);
			*/
		spinWait(_counter, id);

	}
	return;
}


void intra_SpinBarrier(){
	static thread_local SpinBarrier* taskSpinBarrierPtr = nullptr;
	if(taskSpinBarrierPtr==nullptr){
		taskSpinBarrierPtr = TaskManager::getTaskInfo()->spinBarrierObjPtr;
	}
	taskSpinBarrierPtr->wait();
}



}

