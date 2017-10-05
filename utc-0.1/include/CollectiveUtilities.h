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
 * ops include: broadcast, gather, allgather, reduce, allreduce(sum, product, max, min)
 *
 * on one node, every task-thread call these function, but only one thread
 * will do real op, other threads of the same task will return, no wait for
 * real-op completion. so after this function call, user may need call
 * task intra-barrier to ensure the complete the this function in order to
 * get correct data.
 *
 */
#include <typeinfo>
#include <iostream>
#include "mpi.h"
#include "UserTaskBase.h"


template <typename T=char, int localtId = 0>
int TaskBcastBy( UserTaskBase *utask, void *buffer, int count, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Bcast(buffer, count, MPI_CHAR, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Bcast(buffer, count, MPI_INT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Bcast(buffer, count, MPI_LONG, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Bcast(buffer, count, MPI_FLOAT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Bcast(buffer, count, MPI_DOUBLE, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskScatterBy(UserTaskBase *utask, void *sendbuffer, int sendcount, void *recvbuffer, int recvcount, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Scatter(sendbuffer, sendcount, MPI_CHAR, recvbuffer, recvcount, MPI_CHAR, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Scatter(sendbuffer, sendcount, MPI_INT, recvbuffer, recvcount, MPI_INT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Scatter(sendbuffer, sendcount, MPI_LONG, recvbuffer, recvcount, MPI_LONG, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Scatter(sendbuffer, sendcount, MPI_FLOAT, recvbuffer, recvcount, MPI_FLOAT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Scatter(sendbuffer, sendcount, MPI_DOUBLE, recvbuffer, recvcount, MPI_DOUBLE, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskGatherBy(UserTaskBase *utask, void *sendbuffer, int sendcount, void *recvbuffer, int recvcount, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Gather(sendbuffer, sendcount, MPI_CHAR, recvbuffer, recvcount, MPI_CHAR, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Gather(sendbuffer, sendcount, MPI_INT, recvbuffer, recvcount, MPI_INT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Gather(sendbuffer, sendcount, MPI_LONG, recvbuffer, recvcount, MPI_LONG, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Gather(sendbuffer, sendcount, MPI_FLOAT, recvbuffer, recvcount, MPI_FLOAT, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Gather(sendbuffer, sendcount, MPI_DOUBLE, recvbuffer, recvcount, MPI_DOUBLE, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}


template <typename T=char, int localtId = 0>
int TaskAllgatherBy(UserTaskBase *utask, void *sendbuffer, int sendcount, void *recvbuffer, int recvcount){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Allgather(sendbuffer, sendcount, MPI_CHAR, recvbuffer, recvcount, MPI_CHAR, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Allgather(sendbuffer, sendcount, MPI_INT, recvbuffer, recvcount, MPI_INT, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Allgather(sendbuffer, sendcount, MPI_LONG, recvbuffer, recvcount, MPI_LONG, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Allgather(sendbuffer, sendcount, MPI_FLOAT, recvbuffer, recvcount, MPI_FLOAT, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Allgather(sendbuffer, sendcount, MPI_DOUBLE, recvbuffer, recvcount, MPI_DOUBLE, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskReduceSumBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_SUM, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_SUM, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_SUM, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_SUM, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_SUM, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskAllreduceSumBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_SUM, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_SUM, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_SUM, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_SUM, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_SUM, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskReduceProductBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_PROD, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_PROD, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_PROD, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_PROD, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_PROD, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskAllreduceProductBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_PROD, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_PROD, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_PROD, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_PROD, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_PROD, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskReduceMinBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_MIN, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_MIN, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_MIN, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_MIN, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_MIN, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskAllreduceMinBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_MIN, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_MIN, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_MIN, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_MIN, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_MIN, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}


template <typename T=char, int localtId = 0>
int TaskReduceMaxBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count, int root){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_MAX, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_MAX, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_MAX, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_MAX, root, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Reduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_MAX, root, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}

template <typename T=char, int localtId = 0>
int TaskAllreduceMaxBy(UserTaskBase *utask, void *sendbuffer, void *recvbuffer, int count){
	if(utask->__localThreadId != localtId)
		return 0;
	if(typeid(T) == typeid(char)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_CHAR, MPI_MAX, *(utask->__taskComm));
	} else if(typeid(T) == typeid(int)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_INT, MPI_MAX, *(utask->__taskComm));
	} else if(typeid(T) == typeid(long)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_LONG, MPI_MAX, *(utask->__taskComm));
	} else if(typeid(T) == typeid(float)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_FLOAT, MPI_MAX, *(utask->__taskComm));
	} else if(typeid(T) == typeid(double)){
		MPI_Allreduce(sendbuffer, recvbuffer, count, MPI_DOUBLE, MPI_MAX, *(utask->__taskComm));
	} else{
		std::cerr<<"Error: unsupported data type !!!"<<std::endl;
		return 1;
	}
	return 0;
}



#endif /* INCLUDE_COLLECTIVEUTILITIES_H_ */
