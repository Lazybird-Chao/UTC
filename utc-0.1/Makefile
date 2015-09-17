##################
GCC := gcc
G++ := g++
MPICXX := mpicxx

CFLAG :=
CCFLAG := -w -g -std=c++11

###################
MPI_LIB :=	-pthread -Wl,-rpath -Wl,/opt/openmpi/lib -Wl,--enable-new-dtags \
				-L/opt/openmpi/lib -lmpi_cxx -lmpi
BOOST_LIB := /opt/boost/lib/libboost_thread.a \
				/opt/boost/lib/libboost_system.a

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

### test target
obj:= RankList.o RootTask.o \
	  TaskBase.o TaskManager.o \
	  UtcContext.o  UtcMpi.o  
LIB_UTC := libutc.a

	  
test: $(LIB_UTC) main.o
	$(G++)	-o test main.o $(CCFLAG) -lutc $(LIB)
	mkdir -p bin; cd bin; mv ../test ./; mv ../main.o ./
	@echo "make successful !!!"  
main.o: test_main.cc
	$(G++)	-o main.o -c test_main.cc $(CCFLAG) $(INCLUDE) 	



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


###
.PHONY: clean cleanall
clean:
	rm -f ./bin/*.o
	rm -f ./lib/*.o
cleanall:
	rm -rf ./bin
	rm -rf ./lib
