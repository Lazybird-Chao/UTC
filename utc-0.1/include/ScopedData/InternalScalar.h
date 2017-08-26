/*
 * InternalScalar.h
 *
 *  Created on: Dec 5, 2016
 *      Author: Chao
 */

#ifndef INCLUDE_SCOPEDDATA_INTERNALSCALAR_H_
#define INCLUDE_SCOPEDDATA_INTERNALSCALAR_H_

template <typename T>
class InternalScalar{
public:
	virtual operator T()=0;
	virtual T& operator =(T value)=0;
	virtual ~InternalScalar();
};



#endif /* INCLUDE_SCOPEDDATA_INTERNALSCALAR_H_ */
