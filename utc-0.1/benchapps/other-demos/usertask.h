/*
 * Task.h
 *
 */

#ifndef BENCHAPPS_OTHER_DEMOS_USERTASK_H_
#define BENCHAPPS_OTHER_DEMOS_USERTASK_H_

#include <vector>
#include <iostream>
#include "Utc.h"
using namespace iUtc;

template<typename T>
class vectorGen:public UserTaskBase{
private:
	long size;
	T *vectorA;
	T *vectorB;
	std::vector<Conduit *> cdt;
public:
	void initImpl(long size, std::vector<Conduit*> cdt){
		this->size = size;
		this->cdt = cdt;

		vectorA = new T[size];
		vectorB = new T[size];
	}
	void runImpl(int seed, double *time){
		Timer timer;
		timer.start();
		srand(seed);
		T key1 = rand()%size;
		T key2 = rand()%size;
		for(long i=0; i<size; i++){
			vectorA[i] = (key1+i)/size;
			vectorB[i] = (key2+i)/size;
		}
		double t1=timer.stop();
		*time = t1;

		for(int i=0; i<cdt.size(); i++){
			// Write(void* srcData, size_t sizeinbytes, int tag)
			cdt[i]->Write(vectorA, sizeof(T)*size, 0);
			cdt[i]->Write(vectorB, sizeof(T)*size, 1);
		}

	}

	~vectorGen(){
		if(vectorA)
			delete vectorA;
		if(vectorB)
			delete vectorB;
	}
};

enum class Op{
	add,
	sub,
	mul,
	div
};

template<typename T>
class vectorOp: public UserTaskBase{
private:
	long size;
	T *vecA;
	T *vecB;
	T *vecC;
	Conduit *cdtIn,*cdtOut;
public:
	void initImpl(long size, Conduit*cdtIn, Conduit*cdtOut){
		this->size = size;
		vecA = new T[size];
		vecB = new T[size];
		vecC = new T[size];
		this->cdtIn = cdtIn;
		this->cdtOut = cdtOut;

	}
	void runImpl(Op optype, double *time){
		Timer timer;
		// Read(void* dstData, size_t sizeinbytes, int tag)
		cdtIn->Read(vecA, sizeof(T)*size, 0);
		cdtIn->Read(vecB, sizeof(T)*size, 1);
		timer.start();
		int op = 0;
		switch(optype){
		case Op::add:
			op=0;
			for(int j=0; j<100; j++)	//repeat to increase the computation time
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] + vecB[i];
			break;
		case Op::sub:
			op=1;
			for(int j=0; j<100; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] - vecB[i];
			break;
		case Op::mul:
			op=2;
			for(int j=0; j<100; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] * vecB[i];
			break;
		case Op::div:
			op=3;
			for(int j=0; j<100; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] / vecB[i];
			break;
		default:
			break;
		}
		double t1 = timer.stop();
		*time = t1;

		cdtOut->Write(vecC, sizeof(T)*size, 0);
	}

	~vectorOp(){
		if(vecA)
			delete vecA;
		if(vecB)
			delete vecB;
		if(vecC)
			delete vecC;
	}
};

template<typename T>
class vectorMul: public UserTaskBase{
private:
	long size;
	T *vecA;
	T *vecB;
	T result;
	Conduit *cdtInA,*cdtInB;
public:
	void initImpl(int size, Conduit*cdtInA, Conduit*cdtInB){
		this->size = size;
		vecA = new T[size];
		vecB = new T[size];
		this->cdtInA = cdtInA;
		this->cdtInB = cdtInB;

	}
	void runImpl(T *res, double *time){
		Timer timer;
		cdtInA->Read(vecA, sizeof(T)*size, 0);
		cdtInB->Read(vecB, sizeof(T)*size, 0);
		result = 0;
		timer.start();
		for(int j=0; j<100; j++)
		for(long i=0; i<size; i++)
			result += vecA[i]*vecB[i];
		*res = result;
		double t1 = timer.stop();
		*time = t1;
	}

	~vectorMul(){
		if(vecA)
			delete vecA;
		if(vecB)
			delete vecB;
	}
};


#endif /* BENCHAPPS_OTHER_DEMOS_USERTASK_H_ */
