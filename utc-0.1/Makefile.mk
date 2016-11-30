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
    	PrivateScopedData.o \
    	GlobalScopedData.o \
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
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UniqueExeTag.o: UniqueExeTag.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/RootTask.o: RootTask.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskBase.o: TaskBase.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskManager.o: TaskManager.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UtcContext.o:UtcContext.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UtcMpi.o: UtcMpi.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)  
./lib/Task.o: Task.cc Task.inc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskCPU.o: TaskCPU.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/TaskUtilities.o: TaskUtilities.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/UserTaskBase.o: UserTaskBase.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/PrivateScopedData.o: PrivateScopedData.cc PrivateScopedData.inc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/GlobalScopedData.o: GlobalScopedData.cc GlobalScopedData.inc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/ConduitManager.o: ConduitManager.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/Conduit.o: Conduit.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/InprocConduit.o: InprocConduit.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/XprocConduit.o: XprocConduit.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Write.o: InprocConduit_Write.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_BWrite.o: InprocConduit_BWrite.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_PWrite.o: InprocConduit_PWrite.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Read.o: InprocConduit_Read.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/InprocConduit_Async.o: InprocConduit_Async.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)
./lib/XprocConduit_Async.o: XprocConduit_Async.cc
	$(G++)	-o $@ -c $^ $(CCFLAG) $(INCLUDE)	
./lib/Timer.o: Timer.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/Barrier.o: Barrier.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SpinBarrier.o: SpinBarrier.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SharedDataLock.o: SharedDataLock.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
./lib/SpinLock.o: SpinLock.cc
	$(G++)	-o $@ -c $< $(CCFLAG) $(INCLUDE)
	

