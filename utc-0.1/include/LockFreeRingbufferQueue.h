/**
 *
 */
#ifndef LOCK_FREE_RINGBUFFER_QUEUE_
#define LOCK_FREE_RINGBUFFER_QUEUE_

#ifndef __x86_64__
#warning "The program is developed for x86-64 architecture only."
#endif
#if !defined(DCACHE1_LINESIZE) || !DCACHE1_LINESIZE
#ifdef DCACHE1_LINESIZE
#undef DCACHE1_LINESIZE
#endif
#define DCACHE1_LINESIZE 64
#endif
#define ____cacheline_aligned	__attribute__((aligned(DCACHE1_LINESIZE)))

#include <sys/time.h>
#include <limits.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <immintrin.h>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>

#define QUEUE_SIZE	(32 * 1024)
//#define TIMEOUT_COUNT  1000
/*
 * ------------------------------------------------------------------------
 * Naive serialized ring buffer queue
 * ------------------------------------------------------------------------
 */
template<class T, unsigned long Q_SIZE = QUEUE_SIZE>
class NaiveQueue {
private:
	static const unsigned long Q_MASK = Q_SIZE - 1;

public:
	NaiveQueue()
		: head_(0), tail_(0)
	{
		ptr_array_ = (T **)::memalign(getpagesize(),
				Q_SIZE * sizeof(void *));
		assert(ptr_array_);
	}

	void
	push(T *x)
	{
		std::unique_lock<std::mutex> lock(mtx_);

		cond_overflow_.wait(lock, [this]() {
					return tail_ + Q_SIZE > head_;
				});

		ptr_array_[head_++ & Q_MASK] = x;

		cond_empty_.notify_one();
	}

	T *
	pop()
	{
		std::unique_lock<std::mutex> lock(mtx_);

		cond_empty_.wait(lock, [this]() {
					return tail_ < head_;
				});

		T *x = ptr_array_[tail_++ & Q_MASK];

		cond_overflow_.notify_one();

		return x;
	}

private:
	unsigned long		head_, tail_;
	std::condition_variable	cond_empty_;
	std::condition_variable	cond_overflow_;
	std::mutex		mtx_;
	T			**ptr_array_;
};

/*
 * ------------------------------------------------------------------------
 * Boost lock-free fixed size multi-producer multi-consumer queue
 *
 * lockfree: queue - allocate one element more than required
 * the queue internally uses a dummy node, which is not user-visible.
 * therefore we should allocate one element more than needed
 * ------------------------------------------------------------------------
 */
#include "boost/lockfree/queue.hpp"
#include "boost/lockfree/policies.hpp"

template<class T, unsigned long Q_SIZE = QUEUE_SIZE>
class BoostQueue {
public:
	void
	push(T *x)
	{
		while (!q_.push(x))
			asm volatile("rep; nop" ::: "memory");
	}

	T *
	pop()
	{
		T *x;
		while (!q_.pop(x))
			asm volatile("rep; nop" ::: "memory");
		return x;
	}

private:
	boost::lockfree::queue<T *, boost::lockfree::capacity<Q_SIZE+1>> q_;
};


/*
 * ------------------------------------------------------------------------
 * Lock-free N-producers M-consumers ring-buffer queue.
 * ABA problem safe.
 *
 * This implementation is bit complicated, so possibly it has sense to use
 * classic list-based queues. See:
 * 1. D.Fober, Y.Orlarey, S.Letz, "Lock-Free Techniques for Concurrent
 *    Access to Shared Ojects"
 * 2. M.M.Michael, M.L.Scott, "Simple, Fast and Practical Non-Blocking and
 *    Blocking Concurrent Queue Algorithms"
 * 3. E.Ladan-Mozes, N.Shavit, "An Optimistic Approach to Lock-Free FIFO Queues"
 *
 * See also implementation of N-producers M-consumers FIFO and
 * 1-producer 1-consumer ring-buffer from Tim Blechmann:
 *	http://tim.klingt.org/boost_lockfree/
 *	git://tim.klingt.org/boost_lockfree.git
 * 
 * See See Intel 64 and IA-32 Architectures Software Developer's Manual,
 * Volume 3, Chapter 8.2 Memory Ordering for x86 memory ordering guarantees.
 * ------------------------------------------------------------------------
 */

template<class T,
	unsigned long Q_SIZE = QUEUE_SIZE>
class LockFreeQueue {
private:
	static const unsigned long Q_MASK = Q_SIZE - 1;

	struct ThreadLocalPos {
		unsigned long head, tail;
	};

public:
	LockFreeQueue(size_t n_producers, size_t n_consumers)
		: n_producers_(n_producers),
		n_consumers_(n_consumers),
		head_(0),
		tail_(0),
		last_head_(0),
		last_tail_(0)
	{
		auto n = std::max(n_consumers_, n_producers_);
		thr_p_ = (ThreadLocalPos *)memalign(getpagesize(), sizeof(ThreadLocalPos) * n);
		assert(thr_p_);
		// Set per thread tail and head to ULONG_MAX.
		memset((void *)thr_p_, 0xFF, sizeof(ThreadLocalPos) * n);

		ptr_array_ = (T **)memalign(getpagesize(),
				Q_SIZE * sizeof(void *));
		assert(ptr_array_);
	}

	~LockFreeQueue()
	{
		::free(ptr_array_);
		::free(thr_p_);
	}

	// called by each thread to set unique id
	// each thread will use unique local head/tail position
	void setThreadId( unsigned int  id)
	{
		thread_id = id;
	}
	// init queue
	void setupInitPush(){
		head_ = 0;
	}
	int initPush(T* ptr){
		assert(tail_ == 0);
		if(head_ > Q_SIZE)
			return 1;    // push fail
		ptr_array_[head_]=ptr;
		head_++;
		last_head_++;
		return 0;		// push success
	}
	// destroy queue
	void setupEndPop(){
		tail_=0;
	}
	T* endPop(){
		if(tail_ > Q_SIZE)
			return nullptr;		// pop fail
		T* ret = ptr_array_[tail_];
		tail_++;
		return ret;				// pop success
	}

	int
	push(T *ptr)
	{
		/*
		 * Request next place to push.
		 *
		 * Second assignemnt is atomic only for head shift, so there is
		 * a time window in which thr_p_[tid].head = ULONG_MAX, and
		 * head could be shifted significantly by other threads,
		 * so pop() will set last_head_ to head.
		 * After that thr_p_[tid].head is setted to old head value
		 * (which is stored in local CPU register) and written by @ptr.
		 *
		 * First assignment guaranties that pop() sees values for
		 * head and thr_p_[tid].head not greater that they will be
		 * after the second assignment with head shift.
		 *
		 * Loads and stores are not reordered with locked instructions,
		 * se we don't need a memory barrier here.
		 */
		thr_p_[thread_id].head = head_;
		thr_p_[thread_id].head = __sync_fetch_and_add(&head_, 1);

		/*
		 * We do not know when a consumer uses the pop()'ed pointer,
		 * se we can not overwrite it and have to wait the lowest tail.
		 */
		//int timeout_count=0;
		while (__builtin_expect(thr_p_[thread_id].head >= last_tail_ + Q_SIZE, 0))
		{
			auto min = tail_;

			// Update the last_tail_.
			for (size_t i = 0; i < n_consumers_; ++i) {
				auto tmp_t = thr_p_[i].tail;

				// Force compiler to use tmp_h exactly once.
				asm volatile("" ::: "memory");

				if (tmp_t < min)
					min = tmp_t;
			}
			last_tail_ = min;

			if (thr_p_[thread_id].head < last_tail_ + Q_SIZE)
				break;
			/*timeout_count++;
			if(timeout_count>TIMEOUT_COUNT)
				return 1;*/
			_mm_pause();
		}

		ptr_array_[thr_p_[thread_id].head & Q_MASK] = ptr;

		// Allow consumers eat the item.
		thr_p_[thread_id].head = ULONG_MAX;
		return 0;
	}

	// only advance head index, not actual data push
	int push()
	{
		thr_p_[thread_id].head = head_;
		thr_p_[thread_id].head = __sync_fetch_and_add(&head_, 1);

		
		//int timeout_count=0;
		while (__builtin_expect(thr_p_[thread_id].head >= last_tail_ + Q_SIZE, 0))
		{
			auto min = tail_;

			// Update the last_tail_.
			for (size_t i = 0; i < n_consumers_; ++i) {
				auto tmp_t = thr_p_[i].tail;

				// Force compiler to use tmp_h exactly once.
				asm volatile("" ::: "memory");

				if (tmp_t < min)
					min = tmp_t;
			}
			last_tail_ = min;

			if (thr_p_[thread_id].head < last_tail_ + Q_SIZE)
				break;
			/*timeout_count++;
			if(timeout_count>TIMEOUT_COUNT)
				return 1;*/
			_mm_pause();
		}

		// Allow consumers eat the item.
		thr_p_[thread_id].head = ULONG_MAX;
		return 0;
	}

	T *
	pop()
	{
		/*
		 * Request next place from which to pop.
		 * See comments for push().
		 *
		 * Loads and stores are not reordered with locked instructions,
		 * se we don't need a memory barrier here.
		 */
		thr_p_[thread_id].tail = tail_;
		thr_p_[thread_id].tail = __sync_fetch_and_add(&tail_, 1);

		/*
		 * tid'th place in ptr_array_ is reserved by the thread -
		 * this place shall never be rewritten by push() and
		 * last_tail_ at push() is a guarantee.
		 * last_head_ guaraties that no any consumer eats the item
		 * before producer reserved the position writes to it.
		 */
		//int timeout_count =0;
		while (__builtin_expect(thr_p_[thread_id].tail >= last_head_, 0))
		{
			auto min = head_;

			// Update the last_head_.
			for (size_t i = 0; i < n_producers_; ++i) {
				auto tmp_h = thr_p_[i].head;

				// Force compiler to use tmp_h exactly once.
				asm volatile("" ::: "memory");

				if (tmp_h < min)
					min = tmp_h;
			}
			last_head_ = min;

			if (thr_p_[thread_id].tail < last_head_)
				break;
			/*timeout_count++;
			if(timeout_count>TIMEOUT_COUNT)
				return nullptr;*/
			_mm_pause();
		}

		T *ret = ptr_array_[thr_p_[thread_id].tail & Q_MASK];
		// Allow producers rewrite the slot.
		thr_p_[thread_id].tail = ULONG_MAX;
		return ret;
	}

private:
	/*
	 * The most hot members are cacheline aligned to avoid
	 * False Sharing.
	 */

	const size_t n_producers_, n_consumers_;
	// currently free position (next to insert)
	unsigned long	head_ ____cacheline_aligned;
	// current tail, next to pop
	unsigned long	tail_ ____cacheline_aligned;
	// last not-processed producer's pointer
	unsigned long	last_head_ ____cacheline_aligned;
	// last not-processed consumer's pointer
	unsigned long	last_tail_ ____cacheline_aligned;
	ThreadLocalPos		*thr_p_;
	// each thread's index for *thr_p_
	static thread_local unsigned int	thread_id;
	T		**ptr_array_;
};
template<class T,
	unsigned long Q_SIZE>
thread_local unsigned int LockFreeQueue<T, Q_SIZE>::thread_id = -1;

#endif


