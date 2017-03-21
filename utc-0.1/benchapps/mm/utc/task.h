/*
 * task.h
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MM_UTC_TASK_H_
#define BENCHAPPS_MM_UTC_TASK_H_


#include "Utc.h"

template<typename T>
class RandomMatrix:public UserTaskBase{
public:
	void runImpl(T *matrix, int h, int w){
		//matrix = new T[sizeof(T)*w*h];
		T rnumber = (T)(rand()%100)/(rand()%10);
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				matrix[i*w + j] = (j + rnumber)/w;
			}
		}
	}
};



#endif /* BENCHAPPS_MM_UTC_TASK_H_ */
