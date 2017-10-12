/*
 * task.cc
 *
 *  Created on: Oct 11, 2017
 *      Author: Chao
 */

#include "task.h"
#include <iostream>

template<typename T>
void toFile(T* matrix, int h, int w, const char* file, bool isBinary){
		std::ofstream outfile;
		if(isBinary)
			outfile.open(file, std::ios::binary);
		else
			outfile.open(file);
		if(isBinary){
			outfile.write((char*)&h, sizeof(int));
			outfile.write((char*)&w, sizeof(int));
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
template void toFile(float *, int, int, const char*, bool);
template void toFile(double *, int, int, const char*, bool);

template<typename T>
void fromFile(T* &matrix, int& h, int& w, const char* file, bool isBinary){
		std::ifstream infile;
		if(isBinary)
			infile.open(file, std::ios::binary);
		else
			infile.open(file);
		if(isBinary){
			infile.read((char*)&h, sizeof(int));
			infile.read((char*)&w, sizeof(int));
			matrix = new T[w*h];
			infile.read((char*)matrix, h*w*sizeof(T));
		}
		else{
			infile>>h;
			infile>>w;
			matrix = new T[w*h];
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++)
					infile>>matrix[i*w+j];
			}
		}
}
template void fromFile(float*&, int&, int&, const char*, bool);
template void fromFile(double*&, int&, int&, const char*, bool);

template<typename T>
void RandomMatrixGen<T>::runImpl(T **matrix, int *hp, int *wp, const char* filename, bool isBinary){
	if(filename == nullptr){
		int h = *hp;
		int w = *wp;
		*matrix = new T[h*w];
		T rnumber = (T)(rand()%100)/(rand()%10);
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				(*matrix)[i*w + j] = (T)(j + rnumber)/w + i;
			}
		}
	}
	else{
		int h, w;
		T* m;
		fromFile(m, h, w, filename, isBinary);
		*matrix = m;
		*hp = h;
		*wp = w;
	}
}
template class RandomMatrixGen<float>;
template class RandomMatrixGen<double>;


