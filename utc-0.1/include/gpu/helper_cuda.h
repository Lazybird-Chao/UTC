/*
 * helper_cuda.h
 *
 *  Created on: Oct 27, 2016
 *      Author: chao
 */

#ifndef UTC_GPU_HELPER_CUDA_H_
#define UTC_GPU_HELPER_CUDA_H_

#include "UtcGpuBasics.h"

#include "cuda_runtime.h"
#include "cuda.h"
#include "cstdlib"
#include "cstdio"


static const char *_cudartGetErrorEnum(cudaError_t error)
{
    switch (error)
    {

#if CUDA_MAJOR >= 50
        /* Since CUDA 5.0 */
        case cudaSuccess:
			return "cudaSuccess";

		case cudaErrorMissingConfiguration:
			return "cudaErrorMissingConfiguration";

		case cudaErrorMemoryAllocation:
			return "cudaErrorMemoryAllocation";

		case cudaErrorInitializationError:
			return "cudaErrorInitializationError";

		case cudaErrorLaunchFailure:
			return "cudaErrorLaunchFailure";

		case cudaErrorPriorLaunchFailure:
			return "cudaErrorPriorLaunchFailure";

		case cudaErrorLaunchTimeout:
			return "cudaErrorLaunchTimeout";

		case cudaErrorLaunchOutOfResources:
			return "cudaErrorLaunchOutOfResources";

		case cudaErrorInvalidDeviceFunction:
			return "cudaErrorInvalidDeviceFunction";

		case cudaErrorInvalidConfiguration:
			return "cudaErrorInvalidConfiguration";

		case cudaErrorInvalidDevice:
			return "cudaErrorInvalidDevice";

		case cudaErrorInvalidValue:
			return "cudaErrorInvalidValue";

		case cudaErrorInvalidPitchValue:
			return "cudaErrorInvalidPitchValue";

		case cudaErrorInvalidSymbol:
			return "cudaErrorInvalidSymbol";

		case cudaErrorMapBufferObjectFailed:
			return "cudaErrorMapBufferObjectFailed";

		case cudaErrorUnmapBufferObjectFailed:
			return "cudaErrorUnmapBufferObjectFailed";

		case cudaErrorInvalidHostPointer:
			return "cudaErrorInvalidHostPointer";

		case cudaErrorInvalidDevicePointer:
			return "cudaErrorInvalidDevicePointer";

		case cudaErrorInvalidTexture:
			return "cudaErrorInvalidTexture";

		case cudaErrorInvalidTextureBinding:
			return "cudaErrorInvalidTextureBinding";

		case cudaErrorInvalidChannelDescriptor:
			return "cudaErrorInvalidChannelDescriptor";

		case cudaErrorInvalidMemcpyDirection:
			return "cudaErrorInvalidMemcpyDirection";

		case cudaErrorAddressOfConstant:
			return "cudaErrorAddressOfConstant";

		case cudaErrorTextureFetchFailed:
			return "cudaErrorTextureFetchFailed";

		case cudaErrorTextureNotBound:
			return "cudaErrorTextureNotBound";

		case cudaErrorSynchronizationError:
			return "cudaErrorSynchronizationError";

		case cudaErrorInvalidFilterSetting:
			return "cudaErrorInvalidFilterSetting";

		case cudaErrorInvalidNormSetting:
			return "cudaErrorInvalidNormSetting";

		case cudaErrorMixedDeviceExecution:
			return "cudaErrorMixedDeviceExecution";

		case cudaErrorCudartUnloading:
			return "cudaErrorCudartUnloading";

		case cudaErrorUnknown:
			return "cudaErrorUnknown";

		case cudaErrorNotYetImplemented:
			return "cudaErrorNotYetImplemented";

		case cudaErrorMemoryValueTooLarge:
			return "cudaErrorMemoryValueTooLarge";

		case cudaErrorInvalidResourceHandle:
			return "cudaErrorInvalidResourceHandle";

		case cudaErrorNotReady:
			return "cudaErrorNotReady";

		case cudaErrorInsufficientDriver:
			return "cudaErrorInsufficientDriver";

		case cudaErrorSetOnActiveProcess:
			return "cudaErrorSetOnActiveProcess";

		case cudaErrorInvalidSurface:
			return "cudaErrorInvalidSurface";

		case cudaErrorNoDevice:
			return "cudaErrorNoDevice";

		case cudaErrorECCUncorrectable:
			return "cudaErrorECCUncorrectable";

		case cudaErrorSharedObjectSymbolNotFound:
			return "cudaErrorSharedObjectSymbolNotFound";

		case cudaErrorSharedObjectInitFailed:
			return "cudaErrorSharedObjectInitFailed";

		case cudaErrorUnsupportedLimit:
			return "cudaErrorUnsupportedLimit";

		case cudaErrorDuplicateVariableName:
			return "cudaErrorDuplicateVariableName";

		case cudaErrorDuplicateTextureName:
			return "cudaErrorDuplicateTextureName";

		case cudaErrorDuplicateSurfaceName:
			return "cudaErrorDuplicateSurfaceName";

		case cudaErrorDevicesUnavailable:
			return "cudaErrorDevicesUnavailable";

		case cudaErrorInvalidKernelImage:
			return "cudaErrorInvalidKernelImage";

		case cudaErrorNoKernelImageForDevice:
			return "cudaErrorNoKernelImageForDevice";

		case cudaErrorIncompatibleDriverContext:
			return "cudaErrorIncompatibleDriverContext";

		case cudaErrorPeerAccessAlreadyEnabled:
			return "cudaErrorPeerAccessAlreadyEnabled";

		case cudaErrorPeerAccessNotEnabled:
			return "cudaErrorPeerAccessNotEnabled";

		case cudaErrorDeviceAlreadyInUse:
			return "cudaErrorDeviceAlreadyInUse";

		case cudaErrorProfilerDisabled:
			return "cudaErrorProfilerDisabled";

		case cudaErrorProfilerNotInitialized:
			return "cudaErrorProfilerNotInitialized";

		case cudaErrorProfilerAlreadyStarted:
			return "cudaErrorProfilerAlreadyStarted";

		case cudaErrorProfilerAlreadyStopped:
			return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

		case cudaErrorAssert:
			return "cudaErrorAssert";

		case cudaErrorTooManyPeers:
			return "cudaErrorTooManyPeers";

		case cudaErrorHostMemoryAlreadyRegistered:
			return "cudaErrorHostMemoryAlreadyRegistered";

		case cudaErrorHostMemoryNotRegistered:
			return "cudaErrorHostMemoryNotRegistered";
#endif

		case cudaErrorStartupFailure:
			return "cudaErrorStartupFailure";

		case cudaErrorApiFailureBase:
			return "cudaErrorApiFailureBase";
#endif

#if CUDA_MAJOR >= 60
        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
#endif

#if CUDA_MAJOR >= 65
        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
# endif

#if CUDA_MAJOR >= 80
        /* Since CUDA 8.0*/
        case cudaErrorNvlinkUncorrectable :
            return "cudaErrorNvlinkUncorrectable";
#endif
    }

    return "<unknown>";
}


static const char *_cudaGetErrorEnum(CUresult error)
{
    switch (error)
    {
#if CUDA_MAJOR >= 50
    		case CUDA_SUCCESS:
                return "CUDA_SUCCESS";

            case CUDA_ERROR_INVALID_VALUE:
                return "CUDA_ERROR_INVALID_VALUE";

            case CUDA_ERROR_OUT_OF_MEMORY:
                return "CUDA_ERROR_OUT_OF_MEMORY";

            case CUDA_ERROR_NOT_INITIALIZED:
                return "CUDA_ERROR_NOT_INITIALIZED";

            case CUDA_ERROR_DEINITIALIZED:
                return "CUDA_ERROR_DEINITIALIZED";

            case CUDA_ERROR_PROFILER_DISABLED:
                return "CUDA_ERROR_PROFILER_DISABLED";

            case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
                return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

            case CUDA_ERROR_PROFILER_ALREADY_STARTED:
                return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

            case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
                return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

            case CUDA_ERROR_NO_DEVICE:
                return "CUDA_ERROR_NO_DEVICE";

            case CUDA_ERROR_INVALID_DEVICE:
                return "CUDA_ERROR_INVALID_DEVICE";

            case CUDA_ERROR_INVALID_IMAGE:
                return "CUDA_ERROR_INVALID_IMAGE";

            case CUDA_ERROR_INVALID_CONTEXT:
                return "CUDA_ERROR_INVALID_CONTEXT";

            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
                return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

            case CUDA_ERROR_MAP_FAILED:
                return "CUDA_ERROR_MAP_FAILED";

            case CUDA_ERROR_UNMAP_FAILED:
                return "CUDA_ERROR_UNMAP_FAILED";

            case CUDA_ERROR_ARRAY_IS_MAPPED:
                return "CUDA_ERROR_ARRAY_IS_MAPPED";

            case CUDA_ERROR_ALREADY_MAPPED:
                return "CUDA_ERROR_ALREADY_MAPPED";

            case CUDA_ERROR_NO_BINARY_FOR_GPU:
                return "CUDA_ERROR_NO_BINARY_FOR_GPU";

            case CUDA_ERROR_ALREADY_ACQUIRED:
                return "CUDA_ERROR_ALREADY_ACQUIRED";

            case CUDA_ERROR_NOT_MAPPED:
                return "CUDA_ERROR_NOT_MAPPED";

            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
                return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

            case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
                return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

            case CUDA_ERROR_ECC_UNCORRECTABLE:
                return "CUDA_ERROR_ECC_UNCORRECTABLE";

            case CUDA_ERROR_UNSUPPORTED_LIMIT:
                return "CUDA_ERROR_UNSUPPORTED_LIMIT";

            case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
                return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

            case CUDA_ERROR_INVALID_SOURCE:
                return "CUDA_ERROR_INVALID_SOURCE";

            case CUDA_ERROR_FILE_NOT_FOUND:
                return "CUDA_ERROR_FILE_NOT_FOUND";

            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
                return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
                return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

            case CUDA_ERROR_OPERATING_SYSTEM:
                return "CUDA_ERROR_OPERATING_SYSTEM";

            case CUDA_ERROR_INVALID_HANDLE:
                return "CUDA_ERROR_INVALID_HANDLE";

            case CUDA_ERROR_NOT_FOUND:
                return "CUDA_ERROR_NOT_FOUND";

            case CUDA_ERROR_NOT_READY:
                return "CUDA_ERROR_NOT_READY";

            case CUDA_ERROR_LAUNCH_FAILED:
                return "CUDA_ERROR_LAUNCH_FAILED";

            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

            case CUDA_ERROR_LAUNCH_TIMEOUT:
                return "CUDA_ERROR_LAUNCH_TIMEOUT";

            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
                return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

            case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

            case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
                return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

            case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
                return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

            case CUDA_ERROR_CONTEXT_IS_DESTROYED:
                return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

            case CUDA_ERROR_ASSERT:
                return "CUDA_ERROR_ASSERT";

            case CUDA_ERROR_TOO_MANY_PEERS:
                return "CUDA_ERROR_TOO_MANY_PEERS";

            case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

            case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
                return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

            case CUDA_ERROR_UNKNOWN:
                return "CUDA_ERROR_UNKNOWN";
#endif
    }

    return "<unknown>";
}


template< typename T >
void checkrt(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudartGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaRuntimeErrors(val)           checkrt ( (val), #val, __FILE__, __LINE__ )
#define checkCudaDriverErrors(val)            check ( (val), #val, __FILE__, __LINE__ )


// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value)
{
    return (value >= 0 ? (int)(value + 0.5) : (int)(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}


#endif /* UTC_GPU_HELPER_CUDA_H_ */
