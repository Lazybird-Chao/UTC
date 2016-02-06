#ifndef UTC_TASK_UTILITIES_H_
#define UTC_TASK_UTILITIES_H_

namespace iUtc{


// some utility functions can be used in user task code.

// get the process's output file stream
std::ofstream* getProcOstream();

// get the task thread's output file stream
std::ofstream* getThreadOstream();

// return the current thread system id
int getTaskId();

int getParentTaskId();

// return the current thread rank in the task
int getTrank();

// return the current process rank
int getPrank();

// return the current thread rank in current local process
int getLrank();

// return the number of task threads in one process
int getLsize();

// return the number of task threads of the task
int getGsize();

// return the current task obj pointer
TaskBase* getCurrentTask();

TaskBase* getParentTask();

}// end namespace iUtc

#endif
