#ifndef UTC_BASE_H_
#define UTC_BASE_H_

#include "UtcBasics.h"


namespace iUtc{

	class UtcBase
	{
	public:


		virtual int rank()=0;

		virtual void getProcessorName(std::string& name)=0;

		virtual int numProcs() =0;

		virtual void printStatus() =0;

		virtual ~UtcBase(){}


	};
}//namespace iUtc







#endif