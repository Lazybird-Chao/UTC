###################
default:utcruntime

include ./for-dis.comm

###################

utcruntime:
	echo "*******************\n"
	echo "making utc-general library..."
	cd $(PROJECT_HOMEDIR);$(MAKE) -f Make.mk
	echo "*******************\n"
	echo "making utc-gpu library..."
	cd $(PROJECT_HOMEDIR)/src/gpu;$(MAkE) -f Make.mk
	echo "*******************\n"
	echo "making scoped-data library..."
	cd $(PROJECT_HOMEDIR)/src/ScopedData;$(MAKE) -f Make.mk
	echo "*******************\n"
	echo "making internal-shmem library..."
	cd $(PROJECT_HOMEDIR)/src/ScopedData/internal_shmem;$(MAKE) -f Make.mk
	echo "finish !!!\n"

