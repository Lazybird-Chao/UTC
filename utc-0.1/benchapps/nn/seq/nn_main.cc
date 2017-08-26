/*
 * nn_main.cc
 *
 *  Created on: Apr 12, 2017
 *      Author: Chao
 */

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "nn.h"

int main(int argc, char** argv){
	bool printTime = false;
	int numNN, numCoords, numObjs;
	int     isBinaryFile=0;
	char   *filename = nullptr;
	char	*outfile = nullptr;
	float **objects;       /* [numObjs][numCoords] data objects */
	float **objsNN;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"o:i:n:bv"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'i': filename=optarg;
					  break;
			case 'b': isBinaryFile = 1;
					  break;
			case 'n': numNN = atoi(optarg);
					  break;
			case 'o': outfile = optarg;
					  break;
			default:
					  break;
		}
	}

	double t1, t2;

	t1= getTime();
	/* Read input data points from given input file */
	objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
	t2 = getTime();
	double iotime = t2-t1;

	/* target center */
	float *targetObj = new float[numCoords];
	for(int i=0; i<numCoords; i++)
		targetObj[i] = 0;

	objsNN = create_array_2d<float>(numNN, numCoords);
	float *distanceObjs = new float[numObjs];

	t1 = getTime();
	/* compute distance of each objs to the target center*/
	for(int i=0; i<numObjs; i++){
		distanceObjs[i] = sqrtf(euclid_dist_2(numCoords, objects[i], targetObj));
	}
	t2 = getTime();
	double computeTime1 = t2-t1;

	t1= getTime();
	/* find k nearest objs */
	for(int i=0; i<numNN; i++){
		int min = i;
		for(int j=i+1; j<numObjs; j++){
			if(distanceObjs[min]>distanceObjs[j])
				min = j;
		}
		if(min != i){
			float tmp = distanceObjs[i];
			distanceObjs[i] = distanceObjs[min];
			distanceObjs[min] = tmp;
		}
		for(int j=0; j<numCoords; j++)
			objsNN[i][j] = objects[min][j];
	}
	t2= getTime();
	double computeTime2 = t2-t1;

	/* output to file */
	t1 = getTime();
	if(outfile != NULL) {
		int l;
		FILE* fp = fopen(outfile, "w");
		for(int j = 0; j < numNN; j++) {
			fprintf(fp, "Neighbor %d: ", j);
			for(l = 0; l < numCoords; l++)
				fprintf(fp, "%f ", objsNN[j][l]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	t2 = getTime();
	iotime += t2-t1;

	free( objsNN[0]);
	free ( objsNN);
	free(objects[0]);
	free(objects);
	delete distanceObjs;
	delete targetObj;

	std::cout<<"Test complete !!!"<<std::endl;
	if(printTime){
		std::cout<<"Data info:"<<std::endl;
		std::cout<<"\tnumObjs = "<<numObjs<<std::endl;
		std::cout<<"\tnumCoords = "<<numCoords<<std::endl;
		std::cout<<"\tnumNN = "<<numNN<<std::endl;
		std::cout<<"Time info:"<<std::endl;
		std::cout<<"\tcompute 1: "<<std::fixed<<std::setprecision(4)<<computeTime1*1000<<std::endl;
		std::cout<<"\tcompute 2: "<<std::fixed<<std::setprecision(4)<<computeTime2*1000<<std::endl;
		std::cout<<"\tio time: "<<std::fixed<<std::setprecision(4)<<iotime*1000<<std::endl;
	}

	double runtime[3];
	runtime[1] = computeTime1*1000;
	runtime[2] = computeTime2*1000;
	runtime[0] = runtime[1]+runtime[2];
	print_time(3, runtime);

	return 0;
}


















