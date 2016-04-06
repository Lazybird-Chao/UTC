###################

default:libutc.a
###################
##################
PROJECT_HOMEDIR = /home/chao/Desktop/UTC-Project/utc-0.1
PROJECT_BINDIR = $(PROJECT_HOMEDIR)/bin
PROJECT_LIBDIR = $(PROJECT_HOMEDIR)/lib
PROJECT_INCLUDEDIR = $(PROJECT_HOMEDIR)/include
PROJECT_SRCDIR = $(PROJECT_HOMEDIR)/src

##################
GCC = gcc
G++ = g++
MPICXX = mpicxx

CFLAG := -Wall
CCFLAG := -g -std=c++11 -O2

###################
MPI_LIB =	-pthread -Wl,-rpath -Wl,/opt/openmpi/lib -Wl,--enable-new-dtags \
				-L/opt/openmpi/lib -lmpi_cxx -lmpi
BOOST_LIB = /opt/boost/lib/libboost_thread.a \
				/opt/boost/lib/libboost_system.a \
				/opt/boost/lib/libboost_filesystem.a				
LIB = $(MPI_LIB) $(BOOST_LIB)
				
###################		
INCLUDE = -I/opt/openmpi/include
INCLUDE += -I/opt/boost/include
INCLUDE += -I$(PROJECT_HOMEDIR)/include	

###################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)

###################
OBJ_UTC:= 	UtcContext.o UtcMpi.o ProcList.o UniqueExeTag.o\
    		RootTask.o  TaskBase.o TaskManager.o Task.o  \
    		ConduitManager.o Conduit.o InprocConduit.o XprocConduit.o\
    		InprocConduit_Write.o InprocConduit_BWrite.o InprocConduit_PWrite.o	InprocConduit_Read.o \
    		InprocConduit_Async.o XprocConduit_Async.o \
    		Timer.o  Barrier.o SharedDataLock.o
	
### library required object files
libutc.a: $(OBJ_UTC)
	ar -r -uv libutc.a $(OBJ_UTC)
	mkdir -p $(PROJECT_HOMEDIR)/lib; mv *.o $(PROJECT_HOMEDIR)/lib; mv *.a $(PROJECT_HOMEDIR)/lib
	@echo "make library successful !!!" 
ProcList.o: ProcList.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UniqueExeTag.o: UniqueExeTag.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
RootTask.o: RootTask.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskBase.o: TaskBase.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
TaskManager.o: TaskManager.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcContext.o:UtcContext.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
UtcMpi.o: UtcMpi.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)  
Task.o: Task.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
ConduitManager.o: ConduitManager.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
Conduit.o: Conduit.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
InprocConduit.o: InprocConduit.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
XprocConduit.o: XprocConduit.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Write.o: InprocConduit_Write.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_BWrite.o: InprocConduit_BWrite.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_PWrite.o: InprocConduit_PWrite.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Read.o: InprocConduit_Read.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
InprocConduit_Async.o: InprocConduit_Async.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
XprocConduit_Async.o: XprocConduit_Async.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)	
Timer.o: Timer.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
Barrier.o: Barrier.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
SharedDataLock.o: SharedDataLock.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
	

