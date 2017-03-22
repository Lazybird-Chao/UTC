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
	void runImpl(int seed=0){
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
		timer.start();
		for(int i=0; i<cdt.size(); i++){
			cdt[i]->Write(vectorA, sizeof(T)*size, 0);
			cdt[i]->Write(vectorB, sizeof(T)*size, 1);
		}
		double t2 = timer.stop();
		std::cout<<"gen: "<<t1<<" "<<t2<<std::endl;
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
	void runImpl(Op optype){
		Timer timer;
		cdtIn->Read(vecA, sizeof(T)*size, 0);
		cdtIn->Read(vecB, sizeof(T)*size, 1);
		timer.start();
		int op = 0;
		switch(optype){
		case Op::add:
			op=0;
			for(int j=0; j<10; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] + vecB[i];
			break;
		case Op::sub:
			op=1;
			for(int j=0; j<10; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] - vecB[i];
			break;
		case Op::mul:
			op=2;
			for(int j=0; j<10; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] * vecB[i];
			break;
		case Op::div:
			op=3;
			for(int j=0; j<10; j++)
			for(long i=0; i<size; i++)
				vecC[i] = vecA[i] / vecB[i];
			break;
		default:
			break;
		}
		double t1 = timer.stop();
		std::cout<<"Op"<<op<<": "<<t1<<std::endl;

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
	void runImpl(T *res){
		Timer timer;
		cdtInA->Read(vecA, sizeof(T)*size, 0);
		cdtInB->Read(vecB, sizeof(T)*size, 0);
		result = 0;
		timer.start();
		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			result += vecA[i]*vecB[i];
		*res = result;
		double t1 = timer.stop();
		std::cout<<"vmul: "<<t1<<std::endl;
	}

	~vectorMul(){
		if(vecA)
			delete vecA;
		if(vecB)
			delete vecB;
	}
};

template<typename T>
void compare(long size, T res[][2]){
	T *A = new T[size];
	T *B = new T[size];

	T *add = new T[size];
	T *sub = new T[size];
	T *mul = new T[size];
	T *div = new T[size];


	for(int k=0; k<1; k++){
		srand(k);
		T key1 = rand()%size;
		T key2 = rand()%size;
		for(long i=0; i<size; i++){
			A[i] = (key1+i)/size;
			B[i] = (key2+i)/size;
		}

		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			add[i] = A[i]+B[i];

		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			sub[i] = A[i]-B[i];

		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			mul[i] = A[i]*B[i];

		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			div[i] = A[i]/B[i];

		res[k][0]=res[k][1]=0;
		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			res[k][0] += add[i]*mul[i];

		for(int j=0; j<10; j++)
		for(long i=0; i<size; i++)
			res[k][1] += sub[i]*div[i];
	}

	delete A;
	delete B;
	delete add;
	delete sub;
	delete mul;
	delete div;
}

#endif /* BENCHAPPS_OTHER_DEMOS_USERTASK_H_ */
