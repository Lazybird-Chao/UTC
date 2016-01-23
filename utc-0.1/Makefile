##################
GCC := gcc
G++ := g++
MPICXX := mpicxx

CFLAG := -Wall
CCFLAG := -g -std=c++11

###################
MPI_LIB :=	-pthread -Wl,-rpath -Wl,/opt/openmpi/lib -Wl,--enable-new-dtags \
				-L/opt/openmpi/lib -lmpi_cxx -lmpi
BOOST_LIB := /opt/boost/lib/libboost_thread.a \
				/opt/boost/lib/libboost_system.a \
				/opt/boost/lib/libboost_filesystem.a

INCLUDE := -I/opt/openmpi/include
INCLUDE += -I/opt/boost/include
INCLUDE += -I./include

LIB :=
LIB += $(MPI_LIB) $(BOOST_LIB)
LIB += -L./lib
###################
vpath %.h  ./include
vpath %.cc ./src



###################

.PHONY: default
default: test

### utc library targets
obj:= 	UtcContext.o UtcMpi.o RankList.o \
		RootTask.o  TaskBase.o TaskManager.o Task.o  \
		ConduitManager.o Conduit.o InprocConduit.o XprocConduit.o\
		InprocConduit_Write.o InprocConduit_BWrite.o InprocConduit_PWrite.o	InprocConduit_Read.o \
		InprocConduit_Async.o XprocConduit_Async.o \
		Timer.o  Barrier.o SharedDataLock.o
LIB_UTC := libutc.a
 
	
## test target
TARGET_SOURCE:= ./test/bench/latency/utc_latency.cc
TARGET_OBJ:= main.o	
TARGET_EXE:= utc_latency.out

$(TARGET_OBJ): $(TARGET_SOURCE)
	$(G++)	-o $(TARGET_OBJ) -c $(TARGET_SOURCE)  $(CCFLAG) $(INCLUDE) 	


test: $(LIB_UTC) $(TARGET_OBJ)
	$(G++)	-o $(TARGET_EXE) $(TARGET_OBJ) $(CCFLAG) -lutc $(LIB)
	mkdir -p bin; cd bin; mv ../$(TARGET_EXE) ./; mv ../$(TARGET_OBJ) ./
	mkdir -p log
	@echo "make successful !!!" 



### library required object files
$(LIB_UTC): $(obj)
	ar -crv -o libutc.a $(obj)
	mkdir -p lib; cd lib; mv ../*.o ./; mv ../*.a ./
RankList.o: RankList.cc
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
	
	
###
.PHONY: clean cleanall cleanlog
clean:
	rm -rf ./bin/*
	rm -f ./lib/*.o
cleanall:
	rm -rf ./bin
	rm -rf ./lib
	rm -rf ./log
cleanlog:
	rm -rf ./log/*.log