/*
 * PthreadBarrier.h
 *
 *  Created on: Nov 19, 2016
 *      Author: chao
 */

#ifndef UTC_PTHREADBARRIER_H_
#define UTC_PTHREADBARRIER_H_

#include "commonUtil.h"

typedef struct barrier_t barrier_t;
struct barrier_t
{
	unsigned count;
	unsigned total;
	pthread_mutex_t m;
	pthread_cond_t cv;
};

#define BARRIER_FLAG (1UL<<31)
void barrier_destroy(barrier_t *b)
{
	pthread_mutex_lock(&b->m);

	while (b->total > BARRIER_FLAG)
	{
		/* Wait until everyone exits the barrier */
		pthread_cond_wait(&b->cv, &b->m);
	}

	pthread_mutex_unlock(&b->m);

	pthread_cond_destroy(&b->cv);
	pthread_mutex_destroy(&b->m);
}

void barrier_init(barrier_t *b, unsigned count)
{
	pthread_mutex_init(&b->m, NULL);
	pthread_cond_init(&b->cv, NULL);
	b->count = count;
	b->total = BARRIER_FLAG;
}

int barrier_wait(barrier_t *b)
{
	pthread_mutex_lock(&b->m);

	while (b->total > BARRIER_FLAG)
	{
		/* Wait until everyone exits the barrier */
		pthread_cond_wait(&b->cv, &b->m);
	}

	/* Are we the first to enter? */
	if (b->total == BARRIER_FLAG) b->total = 0;

	b->total++;
	//printf("here %d %d\n", b->total, b->count);
	if (b->total == b->count)
	{
		b->total += BARRIER_FLAG - 1;
		pthread_cond_broadcast(&b->cv);

		pthread_mutex_unlock(&b->m);
		//printf("here %d \n", b->total);
		return PTHREAD_BARRIER_SERIAL_THREAD;
	}
	else
	{
		while (b->total < BARRIER_FLAG)
		{
			/* Wait until enough threads enter the barrier */
			pthread_cond_wait(&b->cv, &b->m);
		}

		b->total--;

		/* Get entering threads to wake up */
		if (b->total == BARRIER_FLAG) pthread_cond_broadcast(&b->cv);

		pthread_mutex_unlock(&b->m);

		return 0;
	}
}




#endif /* PTHREADBARRIER_H_ */
