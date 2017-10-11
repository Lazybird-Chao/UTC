/*
 * md5_compute.h
 *
 *      Author: Chao
 */

#ifndef BENCHAPPS_MD5_SEQ_MD5_COMPUTE_H_
#define BENCHAPPS_MD5_SEQ_MD5_COMPUTE_H_

#ifdef HAVE_OPENSSL
#include <openssl/md5.h>
#elif !defined(_MD5_H)
#define _MD5_H

/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD5_u32plus;

typedef struct {
	MD5_u32plus lo, hi;
	MD5_u32plus a, b, c, d;
	unsigned char buffer[64];
	MD5_u32plus block[16];
} MD5_CTX;

void MD5_Init(MD5_CTX *ctx);
void MD5_Update(MD5_CTX *ctx, void *data, unsigned long size);
void MD5_Final(unsigned char *result, MD5_CTX *ctx);

#endif



#endif /* BENCHAPPS_MD5_SEQ_MD5_COMPUTE_H_ */
