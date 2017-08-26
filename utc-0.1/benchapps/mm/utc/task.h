/*
 * task.h
 *
 *  Created on: Mar 20, 2017
 *      Author: Chao
 */

#ifndef BENCHAPPS_MM_UTC_TASK_H_
#define BENCHAPPS_MM_UTC_TASK_H_


#include "Utc.h"
#include <iostream>
#include <fstream>

template<typename T>
class RandomMatrix:public UserTaskBase{
public:
	void runImpl(T *matrix, int h, int w, const char* filename, bool isBinary){
		//matrix = new T[sizeof(T)*w*h];
		if(filename == nullptr){
			T rnumber = (T)(rand()%100)/(rand()%10);
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					matrix[i*w + j] = (j + rnumber)/w;
				}
			}
		}
		else
			fromFile(matrix, h, w, filename, isBinary);
	}

	/*
	 * right now, we cannot overload this runImp() method !!!!
	 * It may cause the using of std::bind() in task.inc file error.
	 */
	/*void runImpl(T *matrix, int h, int w, const char* file, bool isBinary=false){
		fromFile(matrix, h, w, file, isBinary);
	}*/


	static void increaseRowBy(int times, T* &matrix, int h, int w){
		if(times == 1)
			return;
		int newh = h*times;
		T* new_matrix = new T[newh*w];
		for(int i=0; i<times; i++)
			memcpy(new_matrix+h*w*i, matrix, h*w*sizeof(T));
		delete[] matrix;
		matrix = new_matrix;
		return;
	}
	static void decreaseRowBy(int times, T* &matrix, int h, int w){
			if(times == 1)
				return;
			int newh = h/times;
			T* new_matrix = new T[newh*w];
			memcpy(new_matrix, matrix, newh*w*sizeof(T));
			delete[] matrix;
			matrix = new_matrix;
			return;
		}

	static void toFile(T* matrix, int h, int w, const char* file, bool isBinary){
			std::ofstream outfile;
			if(isBinary)
				outfile.open(file, std::ios::binary);
			else
				outfile.open(file);
			if(isBinary){
				outfile<<h<<w;
				outfile.write((char*)matrix, h*w*sizeof(T));
			}
			else{
				outfile<<h<<" "<<w<<std::endl;
				for(int i=0; i<h; i++){
					for(int j=0; j<w; j++)
						outfile<<matrix[i*w+j]<<" ";
					outfile<<std::endl;
				}
			}
	}

	static void fromFile(T* matrix, int h, int w, const char* file, bool isBinary){
			std::ifstream infile;
			if(isBinary)
				infile.open(file, std::ios::binary);
			else
				infile.open(file);
			if(isBinary){
				infile>>h;
				infile>>w;
				infile.read((char*)matrix, h*w*sizeof(T));
			}
			else{
				infile>>h;
				infile>>w;
				for(int i=0; i<h; i++){
					for(int j=0; j<w; j++)
						infile>>matrix[i*w+j];
				}
			}
	}
};



#endif /* BENCHAPPS_MM_UTC_TASK_H_ */
