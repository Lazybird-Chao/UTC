###################
default:utcruntime

include ./for-dis.comm

###################

utcruntime:
	echo "*******************"
	echo "making utc-general library..."
	cd $(PROJECT_HOMEDIR)/src;$(MAKE) -f Make.mk
	echo "*******************"
	echo "making utc-gpu library..."
	cd $(PROJECT_HOMEDIR)/src/gpu;$(MAkE) -f Make.mk
	echo "*******************"
	echo "making scoped-data library..."
	cd $(PROJECT_HOMEDIR)/src/ScopedData;$(MAKE) -f Make.mk
	echo "*******************"
	echo "making internal-shmem library..."
	cd $(PROJECT_HOMEDIR)/src/ScopedData/internal_shmem;$(MAKE) -f Make.mk
	echo "finish !!!"

.PHONY: cleanall
	cd $(PROJECT_HOMEDIR)/src;$(MAKE) clean -f Make.mk
	cd $(PROJECT_HOMEDIR)/src/gpu;$(MAkE) clean -f Make.mk
	cd $(PROJECT_HOMEDIR)/src/ScopedData;$(MAKE) clean -f Make.mk
	cd $(PROJECT_HOMEDIR)/src/ScopedData/internal_shmem;$(MAKE) clean -f Make.mk
