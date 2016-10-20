/*
 * This is a template base class of User task.
 * You can inherit this class when defining your
 * own task class, and realize the necessary member
 * functions and data members.
 */
#ifndef UTC_USER_TASK_BASE_H_
#define UTC_USER_TASK_BASE_H_

#include "PrivateScopedDataBase.h"

#include <vector>
#include <mutex>


class UserTaskBase
{
public:

	UserTaskBase();

	virtual ~UserTaskBase();


	/* necessary member functions */
	virtual void initImpl();

	virtual void runImpl();



	/* other useful member functions */
	void registerPrivateScopedData(iUtc::PrivateScopedDataBase* psData);

	void preInit(int lrank, int trank, int prank, int numLocalThreads,
			int numProcesses, int numTotalThreads);
	void preExit();

	/* useful data members */
	static thread_local int __localThreadId;
	static thread_local int __globalThreadId;
	static thread_local int __processId;
	int __numLocalThreads=0;
	int __numGlobalThreads=0;
	int __numProcesses=0;

private:
	/* other useful member functions */


	/* other useful data members */
	std::vector<iUtc::PrivateScopedDataBase *> __psDataRegistry;
	std::mutex __opMutex;

protected:
	/* other useful member functions */


	/* other useful data members */


};
//static thread_local int __localThreadId;
//static thread_local int __globalThreadId;

#endif
