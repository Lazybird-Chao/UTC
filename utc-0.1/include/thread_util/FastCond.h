/*
 * FastCond.h
 *
 *  Created on: Nov 19, 2016
 *      Author: chao
 */

#ifndef UTC_FASTCOND_H_
#define UTC_FASTCOND_H_

#include "commonUtil.h"
#include "FastMutex.h"


class FastCond{
private:
	struct cv
	{
		FastMutex *m;
		int seq;
		int pad;
	} c;
	typedef cv cv;

public:
	FastCond(){
		c.m = nullptr;
		c.seq = 0;

		return;
	}

	~FastCond(){
		c.m = nullptr;
		c.seq = 0;
	}

	int signal(){
		/* We are waking someone up */
		atomic_add(&(c.seq), 1);

		/* Wake up a thread */
		sys_futex(&(c.seq), FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);

		return 0;
	}

	int broadcast(){
		FastMutex *m = c.m;

		/* No mutex means that there are no waiters */
		if (!m) return 0;

		/* We are waking everyone up */
		atomic_add(&(c.seq), 1);

		/* Wake one thread, and requeue the rest on the mutex */
		sys_futex(&(c.seq), FUTEX_REQUEUE_PRIVATE, 1, (struct timespec *) INT_MAX, m, 0);

		return 0;
	}

	int wait(FastMutex *m){
		int seq = c.seq;

		if (c.m != m)
		{
			/* Atomically set mutex inside cv */
			cmpxchg(&(c.m), NULL, (unsigned long long)m);
			if (c.m != m) return EINVAL;
		}

		m->unlock();

		sys_futex(&(c.seq), FUTEX_WAIT_PRIVATE, seq, NULL, NULL, 0);

		while (xchg_32(&(m->m.b.locked), 257) & 1)
		{
			sys_futex(m, FUTEX_WAIT_PRIVATE, 257, NULL, NULL, 0);
		}

		return 0;
	}

};





#endif /* FASTCOND_H_ */
