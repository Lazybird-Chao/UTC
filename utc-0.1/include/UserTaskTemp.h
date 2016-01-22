/*
 * This is a template base class of User task.
 * You can inherit this class when defining your
 * own task class, and realize the necessary member
 * functions and data members.
 */

class UserTaskBase
{
public:

	UserTaskBase(){}
		// you can define or overload a meaning full constructor in derived class
	~UserTaskBase(){}
		// you should clean the used data and free allocated memory

	/* necessary member functions */
	void init(){std::cerr<<"Using the template User task class!\n";}
		// usually initialize data members and prepare for computation,
		// you should implement this method and can overload with arguments
		// in derived class
	void run(){std::cerr<<"Using the template User task class!\n";}
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

protected:
	/* other useful member functions */


	/* other useful data members */


};
