#ifndef UTC_TASKARTIFACT_H_
#define UTC_TASKARTIFACT_H_

namespace iUtc{

class TaskArtifact{
public:
	virtual int launchThreads()=0;

	virtual int initImpl()=0;

	virtual int runImpl()=0;

	virtual int execImpl()=0;

	virtual int waitImpl()=0;

	virtual int finishImpl()=0;

	virtual ~TaskArtifact(){};

private:

};




}


#endif
