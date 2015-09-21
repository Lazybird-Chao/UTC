################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp 

CC_SRCS += \
../src/RankList.cc \
../src/RootTask.cc \
../src/TaskBase.cc \
../src/TaskManager.cc \
../src/UtcContext.cc \
../src/UtcMpi.cc 

OBJS += \
./src/RankList.o \
./src/RootTask.o \
./src/TaskBase.o \
./src/TaskManager.o \
./src/UtcContext.o \
./src/UtcMpi.o \
./src/main.o 

CC_DEPS += \
./src/RankList.d \
./src/RootTask.d \
./src/TaskBase.d \
./src/TaskManager.d \
./src/UtcContext.d \
./src/UtcMpi.d 

CPP_DEPS += \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


