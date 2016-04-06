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

LINKFLAG =

###################
MPI_LIB =	-pthread -Wl,-rpath -Wl,/opt/openmpi/lib -Wl,--enable-new-dtags \
				-L/opt/openmpi/lib -lmpi_cxx -lmpi
BOOST_LIB = /opt/boost/lib/libboost_thread.a \
				/opt/boost/lib/libboost_system.a \
				/opt/boost/lib/libboost_filesystem.a
				
LIB = $(PROJECT_LIBDIR)/libutc.a
LIB += $(MPI_LIB) $(BOOST_LIB)
				
###################		
INCLUDE = -I/opt/openmpi/include
INCLUDE += -I/opt/boost/include
INCLUDE += -I$(PROJECT_HOMEDIR)/include		


####################
LIB_UTC = libutc

$(LIB_UTC):
	cd $(PROJECT_HOMEDIR); $(MAKE)


#####################
.PHONY: cleanlib cleanlog

cleanlib:
	rm -rf $(PROJECT_BINDIR)
	rm -rf $(PROJECT_LIDDIR)
cleanlog:
	rm -rf $(PROJECT_HOMEDIR)/log	