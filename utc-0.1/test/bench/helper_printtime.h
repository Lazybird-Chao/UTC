#ifndef _HELPER_PRINTTIME_H_
#define _HELPER_PRINTTIME_H_

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






#endif

