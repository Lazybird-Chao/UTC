###################
default:libutc.a

include ./make.comm

###################
vpath %.h  $(PROJECT_INCLUDEDIR)
vpath %.inc $(PROJECT_INCLUDEDIR)
vpath %.cc $(PROJECT_SRCDIR)

###################
OBJS:= 	UtcContext.o \
		UtcMpi.o \
		ProcList.o \
		UniqueExeTag.o \
    	RootTask.o  \
    	TaskBase.o  \
    	TaskManager.o \
    	Task.o  \
    	TaskCPU.o \
    	TaskUtilities.o \
    	UserTaskBase.o \
		ConduitManager.o \
		Conduit.o \
		InprocConduit.o \
		XprocConduit.o \
		InprocConduit_Write.o \
		InprocConduit_BWrite.o \
		InprocConduit_PWrite.o	\
		InprocConduit_Read.o \
		InprocConduit_Async.o \
		XprocConduit_Async.o \
		Timer.o  \
		Barrier.o \
		SpinBarrier.o \
		SharedDataLock.o \
		SpinLock.o
		
OBJ_UTC = $(addprefix ./lib/, $(OBJS))

### library required object files
libutc.a: $(OBJ_UTC)
	ar -r -uv ./lib/libutc.a $(OBJ_UTC)
	@echo "make library successful !!!" 
./lib/ProcList.o: ProcList.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UniqueExeTag.o: UniqueExeTag.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/RootTask.o: RootTask.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskBase.o: TaskBase.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskManager.o: TaskManager.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UtcContext.o:UtcContext.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UtcMpi.o: UtcMpi.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)  
./lib/Task.o: Task.cc Task.inc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskCPU.o: TaskCPU.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskUtilities.o: TaskUtilities.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UserTaskBase.o: UserTaskBase.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/ConduitManager.o: ConduitManager.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/Conduit.o: Conduit.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/InprocConduit.o: InprocConduit.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/XprocConduit.o: XprocConduit.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Write.o: InprocConduit_Write.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_BWrite.o: InprocConduit_BWrite.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_PWrite.o: InprocConduit_PWrite.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Read.o: InprocConduit_Read.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Async.o: InprocConduit_Async.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/XprocConduit_Async.o: XprocConduit_Async.cc
	$(C++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)	
./lib/Timer.o: Timer.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/Barrier.o: Barrier.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SpinBarrier.o: SpinBarrier.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SharedDataLock.o: SharedDataLock.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SpinLock.o: SpinLock.cc
	$(C++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
	

