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

#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "nn.h"

int main(int argc, char** argv){
	bool printTime = false;
	int numNN, numCoords, numObjs;
	int     isBinaryFile;
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
			if(distanceObjs[min]<distanceObjs[j])
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


	if(printTime){
		std::cout<<"Data info:"<<std::endl;
		std::cout<<"\tnumObjs = "<<numObjs<<std::endl;
		std::cout<<"\tnumCoords = "<<numCoords<<std::endl;
		std::cout<<"\numNN = "<<numNN<<std::endl;
		std::cout<<"Time info:"<<std::endl;
		std::cout<<"compute 1: "<<std::fixed<<std::setprecision(4)<<computeTime1<<std::endl;
		std::cout<<"compute 2: "<<std::fixed<<std::setprecision(4)<<computeTime2<<std::endl;
		std::cout<<"io time: "<<std::fixed<<std::setprecision(4)<<iotime<<std::endl;
	}

	double runtime[2];
	runtime[0] = computeTime1;
	runtime[1] = computeTime2;
	//print_time(2, runtime);

	return 0;
}


















