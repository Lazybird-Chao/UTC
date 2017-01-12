/*
 * CollectiveUtilities.h
 *
 *  Created on: Jan 11, 2017
 *      Author: Chao
 *
 *
 */

#ifndef INCLUDE_COLLECTIVEUTILITIES_H_
#define INCLUDE_COLLECTIVEUTILITIES_H_


/*
 * here we implement collective functions to op for a task which span
 * multiple nodes and collect data blocks which are on different nodes
 * to somewhere.
 *
 * ops include: broadcast, gather, allgather, reduce, allreduce
 *
 * on one node, every task-thread call these function, but only one thread
 * will do real op, other threads of the same task will return, no wait for
 * real-op completion. so after this function call, user may need call
 * task intra-barrier to ensure the complete the this function in order to
 * get correct data.
 *
 */





#endif /* INCLUDE_COLLECTIVEUTILITIES_H_ */
