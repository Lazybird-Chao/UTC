#ifndef UTC_TASKARTIFACT_H_
#define UTC_TASKARTIFACT_H_

#include <functional>

namespace iUtc{

class TaskArtifact{
public:
	virtual void launchThreads(){};

	virtual int initImpl(std::function<void()> initHandle)=0;

	virtual int runImpl(std::function<void()> runHandle)=0;

	virtual int execImpl(std::function<void()> execHandle)=0;

	virtual int waitImpl()=0;

	virtual int finishImpl()=0;

	virtual bool hasActiveLocalThread()=0;
	virtual void waitLocalThreadFinish()=0;

	virtual void updateUserTaskObj(UserTaskBase* newUserObj){};

	virtual ~TaskArtifact(){};

private:

};




}


#endif
