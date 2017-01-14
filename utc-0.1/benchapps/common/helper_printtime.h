#ifndef _HELPER_PRINTTIME_H_
#define _HELPER_PRINTTIME_H_

#ifdef __cplusplus
#include <fstream>
#include <iostream>
#include <iomanip>

const char *filename ="./time_record.txt";
void print_time(int timerCount, double* timeVal){
	std::fstream timeRecord;
	timeRecord.open(filename, std::ios::in);

	double *totalTimeVal = (double*)malloc(sizeof(double)*timerCount);
	double *lineTimeVal[100];
	for(int i=0; i<timerCount; i++){
		totalTimeVal[i]=0.0;
	}
	//std::cout<<timeRecord.is_open()<<std::endl;
	int line=0;
	if(timeRecord.is_open()){
		char c;
		timeRecord.get(c);
		char lineend[100];
		while(c != 'A'){
			timeRecord.seekg(-1,std::ios::cur);
			lineTimeVal[line]=(double*)malloc(sizeof(double)*timerCount);
			for(int i=0; i<timerCount; i++){
				timeRecord>>lineTimeVal[line][i];
				totalTimeVal[i]+=lineTimeVal[line][i];
			}
			timeRecord.getline(lineend, 100);
			line++;
			timeRecord.get(c);
		}
		timeRecord.close();
	}

	timeRecord.open(filename, std::ios::out);
	//std::cout<<timeRecord.is_open()<<std::endl;
	for(int i=0; i<line; i++){
		for(int j=0; j<timerCount; j++){
			timeRecord<<std::fixed<<std::setprecision(4)<<std::setw(10)<<lineTimeVal[i][j];
		}
		timeRecord<<std::endl;
	}
	for(int j=0; j<timerCount; j++){
		//std::cout<<j;
		timeRecord<<std::fixed<<std::setprecision(4)<<std::setw(10)<<timeVal[j];
		totalTimeVal[j]+=timeVal[j];
	}
	timeRecord<<std::endl;
	line++;
	timeRecord<<"Average:\n";
	//std::cout<<line<<std::endl;
	for(int i=0; i<timerCount; i++){
		totalTimeVal[i]/= line;
	}
	for(int i=0; i<timerCount; i++){
		timeRecord<<std::fixed<<std::setprecision(4)<<std::setw(10)<<totalTimeVal[i];
	}
	timeRecord<<std::endl<<std::endl;

	timeRecord.close();


}

#else

#include <stdio.h>
const char *filename ="./time_record.txt";
void print_time(int timerCount, double* timeVal){
	FILE *timeRecord;
	timeRecord=fopen(filename,"r");
	double *totalTimeVal = (double*)malloc(sizeof(double)*timerCount);
		double *lineTimeVal[100];
		for(int i=0; i<timerCount; i++){
			totalTimeVal[i]=0.0;
		}
		//std::cout<<timeRecord.is_open()<<std::endl;
		int line=0;
		if(timeRecord){
			char c;
			c=fgetc(timeRecord);
			char lineend[100];
			while(c != 'A'){
				fseek(timeRecord, -1, SEEK_CUR);
				lineTimeVal[line]=(double*)malloc(sizeof(double)*timerCount);
				for(int i=0; i<timerCount; i++){
					fscanf(timeRecord, "%lf",&lineTimeVal[line][i]);
					totalTimeVal[i]+=lineTimeVal[line][i];
				}
				//fgets(lineend, 100, timeRecord);
				line++;
				c=fgetc(timeRecord);
			}
			fclose(timeRecord);
		}

		timeRecord=fopen(filename, "w");
			//std::cout<<timeRecord.is_open()<<std::endl;
			for(int i=0; i<line; i++){
				for(int j=0; j<timerCount; j++){
					fprintf(timeRecord, "%.4lf\t\t",lineTimeVal[i][j]);
				}
				fprintf(timeRecord, "\n");
			}
			for(int j=0; j<timerCount; j++){
				//std::cout<<j;
				fprintf(timeRecord, "%.4lf\t\t",timeVal[j]);
				totalTimeVal[j]+=timeVal[j];
			}
			fprintf(timeRecord, "\n");
			line++;
			fprintf(timeRecord, "Average\n");
			//std::cout<<line<<std::endl;
			for(int i=0; i<timerCount; i++){
				totalTimeVal[i]/= line;
			}
			for(int i=0; i<timerCount; i++){
				fprintf(timeRecord, "%.4lf\t\t",totalTimeVal[i]);
			}
			fprintf(timeRecord, "\n");

			fclose(timeRecord);

}


#endif





#endif

