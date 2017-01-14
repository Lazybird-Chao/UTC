/*
 * helper_check.h
 *
 *  Created on: Jan 13, 2017
 *      Author: chao
 */

#ifndef HELPER_CHECK_H_
#define HELPER_CHECK_H_

#define checkCudaErr(_err_call)		{\
									cudaError_t _err = _err_call;\
									if(_err != cudaSuccess){\
									std::cout<<"cuda err at: "<<__LINE__<<\
									", "<<__FILE__<<\
									" \""<<cudaGetErrorString(_err)<<"\""<<\
									std::endl;\
									}}


#endif /* HELPER_CHECK_H_ */
