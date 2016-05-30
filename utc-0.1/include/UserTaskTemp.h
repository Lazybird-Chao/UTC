/*
 * This is a template base class of User task.
 * You can inherit this class when defining your
 * own task class, and realize the necessary member
 * functions and data members.
 */

#include "TaskUtilities.h"

class UserTaskTMP
{
public:

	UserTaskTMP(){
		__localThreadId = iUtc::getTrank();
		__globalThreadId = iUtc::getPrank();
		__numLocalThreads = iUtc::getLsize();
		__numGlobalThreads = iUtc::getGsize();
	}
		// you can define or overload a meaning full constructor in derived class
	virtual ~UserTaskTMP(){}
		// you should clean the used data and free allocated memory

	/* necessary member functions */
	virtual void initImpl(){std::cerr<<"Using the template User task class!\n";}
		// usually initialize data members and prepare for computation,
		// you should implement this method and can overload with arguments
		// in derived class
	virtual void runImpl(){std::cerr<<"Using the template User task class!\n";}
		// usually implement the main algorithm, doing computation and communication,
		// you should implement this method in derived class


	/* other useful member functions */



	/* useful data members */
		// data members are shared by all threads of a task object in one process,
		// be careful when accessing data members in different running task-threads

private:
	/* other useful member functions */


	/* other useful data members */
		// e.g. Conduit* cdt;
		//      initialize cdt in init() and use it for communication in run()
	static thread_local int __localThreadId=-1;
	static thread_local int __globalThreadId=-1;
	int __numLocalThreads=0;
	int __numGlobalThreads=0;

protected:
	/* other useful member functions */


	/* other useful data members */


};
//static thread_local int __localThreadId;
//static thread_local int __globalThreadId;
