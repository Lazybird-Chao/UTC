obj = kmeans_main.o kmeans_compute.o file_io.o
bin = kmeans
default: $(bin)

C++ = mpicxx
CCFLAGS = -O2  -std=c++11
CCFLAGS += -Wno-deprecated -Wno-write-strings
LIB = -lrt -lm

DEBUG = 0
ifeq ($(DEBUG), 1)
	CCFLAGS += -g
endif

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
  
kmeans_main.o:kmeans_main.cc kmeans.h
	$(C++) -o $@ $(CCFLAGS) -c $<
    
kmeans_compute.o:kmeans_compute.cc kmeans.h
	$(C++) -o $@ $(CCFLAGS) -c $<

file_io.o:file_io.cc kmeans.h
	$(C++) -o $@ $(CCFLAGS) -c $<

$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
