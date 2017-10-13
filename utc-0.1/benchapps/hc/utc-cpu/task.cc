/*
 * task.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: chaoliu
 */

#include "task.h"
#include <cstdlib>
#include <cstdio>

template<typename T>
void Output<T>::runImpl(T *buffer, int w, int h, char *ofile){
	FILE *fp = fopen(ofile, "w");
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			fprintf(fp, "%.5f ", buffer[i*w +j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

template class Output<float>;
template class Output<double>;


