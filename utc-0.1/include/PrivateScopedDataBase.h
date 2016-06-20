/*
 * PrivateScopedDataBase.h
 *
 *  Created on: Jun 20, 2016
 *      Author: chaoliu
 */

#ifndef UTC_PRIVATESCOPEDDATABASE_H_
#define UTC_PRIVATESCOPEDDATABASE_H_

namespace iUtc{
class PrivateScopedDataBase{
public:
	virtual void init(){};
	virtual void destroy(){};

	virtual ~PrivateScopedDataBase(){};
};


}



#endif /* UTC_PRIVATESCOPEDDATABASE_H_ */
