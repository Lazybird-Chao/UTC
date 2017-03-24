obj = md5_main.o md5_compute.o
bin = md5_main
default: $(bin)

C++ = g++
CCFLAGS = -O2 -g -std=c++11
LIB = -lm -lrt

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	
%.o:%.cc
	$(C++) -o $@ $(CCFLAGS) -c $<

md5_main.o: md5_main.cc md5_main.h md5_compute.h

md5_compute.o: md5_compute.cc md5_compute.h


$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)