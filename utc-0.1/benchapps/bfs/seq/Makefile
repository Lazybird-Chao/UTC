obj = bfs_main.o
bin = bfs_main
default: $(bin)

C++ = g++
CCFLAGS = -O2 -g -std=c++11
LIB = -lm

COMMONDIR = ../../common
helperObj = helperObj.o
$(helperObj):$(COMMONDIR)/helper_getopt.c
	$(C++) -o $@  -c $<
	
%.o:%.cc
	$(C++) -o $@ $(CCFLAGS) -c $<

bfs_main.o: bfs_main.cc bfs_main.h



$(bin): $(obj) $(helperObj)
	$(C++) -o $@ $(CCFLAGS) $(LIB) $(obj) $(helperObj)

.PHONY: clean
clean:
	rm -f $(obj) $(bin) $(helperObj)
	