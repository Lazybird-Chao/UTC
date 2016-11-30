/*
 * FastMutex.h
 *
 *  Created on: Nov 19, 2016
 *      Author: chao
 */

#ifndef UTC_FASTMUTEX_H_
#define UTC_FASTMUTEX_H_

#include "commonUtil.h"

class FastMutex{
private:
	union mutex
	{
		unsigned u;
		struct
		{
			unsigned char locked;
			unsigned char contended;
		} b;
	} m;
	typedef union mutex mutex;

public:
	friend class FastCond;
	FastMutex(){
		m.u = 0;
		return;
	}

	~FastMutex(){
		m.u = 0;
		return;
	}

	int lock(){
		int i;

		/* Try to grab lock */
		for (i = 0; i < 100; i++)
		{
			if (!xchg_8(&(m.b.locked), 1)) return 0;

			cpu_relax();
		}

		/* Have to sleep */
		while (xchg_32(&(m.u), 257) & 1)
		{
			sys_futex(&m, FUTEX_WAIT_PRIVATE, 257, NULL, NULL, 0);
		}

		return 0;
	}

	int unlock(){
		int i;

			/* Locked and not contended */
			if ((m.u == 1) && (cmpxchg(&(m.u), 1, 0) == 1)) return 0;

			/* Unlock */
			m.b.locked = 0;

			mem_barrier();

			/* Spin and hope someone takes the lock */
			for (i = 0; i < 200; i++)
			{
				if (*(volatile char *) &m.b.locked) return 0;

				cpu_relax();
			}

			/* We need to wake someone up */
			m.b.contended = 0;

			sys_futex(&m, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);

			return 0;
	}

	int trylock(){
		unsigned c = xchg_8(&(m.b.locked), 1);
		if (!c) return 0;
		return EBUSY;
	}

};




#endif /* FASTMUTEX_H_ */
