/*
 * md5_kernel.cc
 *
 *      Author: Chao
 */

#include "md5_kernel.h"
#include "stdio.h"

/*
 * The basic MD5 functions.
 *
 * F and G are optimized compared to their RFC 1321 definitions for
 * architectures that lack an AND-NOT instruction, just like in Colin Plumb's
 * implementation.
 */
#define F(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)			((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)			((x) ^ (y) ^ (z))
#define I(x, y, z)			((y) ^ ((x) | ~(z)))

/*
 * The MD5 transformation for all four rounds.
 */
#define STEP(f, a, b, c, d, x, t, s) \
	(a) += f((b), (c), (d)) + (x) + (t); \
	(a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
	(a) += (b);

/*
#define SET(n) \
	(*(MD5_u32plus *)&ptr[(n) * 4])
#define GET(n) \
	SET(n)
*/
#define SET(n, _p, _off) \
		(*(MD5_u32plus *)&_p[n*_off])
#define GET(n, _p, _off) \
	SET(n, _p, _off)


/*
 * here we use one cuda thread to deal with one input buffer.
 * This buffer's data will be processed with MD5 procedures sequentially.
 *
 */
__global__ void md5_process(
		uint8_t *inputs,
		uint8_t *out,
		long numbuffs,
		long buffsize){

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int idx = bx*blockDim.x + tx;
	//the thread need to process a buffer
	if(idx < numbuffs){
		MD5_CTX context;
		MD5_Init(&context);
		MD5_Update(&context, &inputs[idx], numbuffs, buffsize);
		//MD5_CTX *ctx = &context;
		//printf("%d: %d %d %d %d \n", threadIdx.x, ctx->a, ctx->b, ctx->c, ctx->d);
		MD5_Final((unsigned char*)&out[idx], &context, numbuffs);
	}


}


/*
 * This processes one or more 64-byte data blocks, but does NOT update
 * the bit counters.  There are no alignment requirements.
 */
__device__ void *body_with_global(MD5_CTX *ctx, void *data, long offset, unsigned long size)
{
	unsigned char *ptr;
	MD5_u32plus a, b, c, d;
	MD5_u32plus saved_a, saved_b, saved_c, saved_d;

	ptr = (unsigned char *)data;

	a = ctx->a;
	b = ctx->b;
	c = ctx->c;
	d = ctx->d;

	// each thread has a 64 Bytes data_block on shared memory
	// assume 256 threads per block, change if needed
#define __blocksize 256
	__shared__ unsigned int data_block[16*__blocksize];
	unsigned int* ptr_s = &data_block[threadIdx.x];

	do {
		/*if(threadIdx.x ==0){
			printf("new round: ");
		}*/
		/* load 64 Bytes to data_block */
		for(int i=0; i<16; i++){
			unsigned int tmp;
			tmp = ptr[(i*4+3)*offset];
			tmp = (tmp<<8) + ptr[(i*4+2)*offset];
			tmp = (tmp<<8) + ptr[(i*4+1)*offset];
			tmp = (tmp<<8) + ptr[i*4*offset];
			data_block[i*__blocksize+threadIdx.x] = tmp;
			/*if(threadIdx.x ==0){
				printf("%u ", tmp);
			}*/
		}
		/*if(threadIdx.x ==0){
			printf("\n");
		}*/
		__syncthreads();

		saved_a = a;
		saved_b = b;
		saved_c = c;
		saved_d = d;

/* Round 1 */
		STEP(F, a, b, c, d, SET(0, ptr_s, __blocksize), 0xd76aa478, 7)
		STEP(F, d, a, b, c, SET(1, ptr_s, __blocksize), 0xe8c7b756, 12)
		STEP(F, c, d, a, b, SET(2, ptr_s, __blocksize), 0x242070db, 17)
		STEP(F, b, c, d, a, SET(3, ptr_s, __blocksize), 0xc1bdceee, 22)
		STEP(F, a, b, c, d, SET(4, ptr_s, __blocksize), 0xf57c0faf, 7)
		STEP(F, d, a, b, c, SET(5, ptr_s, __blocksize), 0x4787c62a, 12)
		STEP(F, c, d, a, b, SET(6, ptr_s, __blocksize), 0xa8304613, 17)
		STEP(F, b, c, d, a, SET(7, ptr_s, __blocksize), 0xfd469501, 22)
		STEP(F, a, b, c, d, SET(8, ptr_s, __blocksize), 0x698098d8, 7)
		STEP(F, d, a, b, c, SET(9, ptr_s, __blocksize), 0x8b44f7af, 12)
		STEP(F, c, d, a, b, SET(10, ptr_s, __blocksize), 0xffff5bb1, 17)
		STEP(F, b, c, d, a, SET(11, ptr_s, __blocksize), 0x895cd7be, 22)
		STEP(F, a, b, c, d, SET(12, ptr_s, __blocksize), 0x6b901122, 7)
		STEP(F, d, a, b, c, SET(13, ptr_s, __blocksize), 0xfd987193, 12)
		STEP(F, c, d, a, b, SET(14, ptr_s, __blocksize), 0xa679438e, 17)
		STEP(F, b, c, d, a, SET(15, ptr_s, __blocksize), 0x49b40821, 22)

/* Round 2 */
		STEP(G, a, b, c, d, GET(1, ptr_s, __blocksize), 0xf61e2562, 5)
		STEP(G, d, a, b, c, GET(6, ptr_s, __blocksize), 0xc040b340, 9)
		STEP(G, c, d, a, b, GET(11, ptr_s, __blocksize), 0x265e5a51, 14)
		STEP(G, b, c, d, a, GET(0, ptr_s, __blocksize), 0xe9b6c7aa, 20)
		STEP(G, a, b, c, d, GET(5, ptr_s, __blocksize), 0xd62f105d, 5)
		STEP(G, d, a, b, c, GET(10, ptr_s, __blocksize), 0x02441453, 9)
		STEP(G, c, d, a, b, GET(15, ptr_s, __blocksize), 0xd8a1e681, 14)
		STEP(G, b, c, d, a, GET(4, ptr_s, __blocksize), 0xe7d3fbc8, 20)
		STEP(G, a, b, c, d, GET(9, ptr_s, __blocksize), 0x21e1cde6, 5)
		STEP(G, d, a, b, c, GET(14, ptr_s, __blocksize), 0xc33707d6, 9)
		STEP(G, c, d, a, b, GET(3, ptr_s, __blocksize), 0xf4d50d87, 14)
		STEP(G, b, c, d, a, GET(8, ptr_s, __blocksize), 0x455a14ed, 20)
		STEP(G, a, b, c, d, GET(13, ptr_s, __blocksize), 0xa9e3e905, 5)
		STEP(G, d, a, b, c, GET(2, ptr_s, __blocksize), 0xfcefa3f8, 9)
		STEP(G, c, d, a, b, GET(7, ptr_s, __blocksize), 0x676f02d9, 14)
		STEP(G, b, c, d, a, GET(12, ptr_s, __blocksize), 0x8d2a4c8a, 20)

/* Round 3 */
		STEP(H, a, b, c, d, GET(5, ptr_s, __blocksize), 0xfffa3942, 4)
		STEP(H, d, a, b, c, GET(8, ptr_s, __blocksize), 0x8771f681, 11)
		STEP(H, c, d, a, b, GET(11, ptr_s, __blocksize), 0x6d9d6122, 16)
		STEP(H, b, c, d, a, GET(14, ptr_s, __blocksize), 0xfde5380c, 23)
		STEP(H, a, b, c, d, GET(1, ptr_s, __blocksize), 0xa4beea44, 4)
		STEP(H, d, a, b, c, GET(4, ptr_s, __blocksize), 0x4bdecfa9, 11)
		STEP(H, c, d, a, b, GET(7, ptr_s, __blocksize), 0xf6bb4b60, 16)
		STEP(H, b, c, d, a, GET(10, ptr_s, __blocksize), 0xbebfbc70, 23)
		STEP(H, a, b, c, d, GET(13, ptr_s, __blocksize), 0x289b7ec6, 4)
		STEP(H, d, a, b, c, GET(0, ptr_s, __blocksize), 0xeaa127fa, 11)
		STEP(H, c, d, a, b, GET(3, ptr_s, __blocksize), 0xd4ef3085, 16)
		STEP(H, b, c, d, a, GET(6, ptr_s, __blocksize), 0x04881d05, 23)
		STEP(H, a, b, c, d, GET(9, ptr_s, __blocksize), 0xd9d4d039, 4)
		STEP(H, d, a, b, c, GET(12, ptr_s, __blocksize), 0xe6db99e5, 11)
		STEP(H, c, d, a, b, GET(15, ptr_s, __blocksize), 0x1fa27cf8, 16)
		STEP(H, b, c, d, a, GET(2, ptr_s, __blocksize), 0xc4ac5665, 23)

/* Round 4 */
		STEP(I, a, b, c, d, GET(0, ptr_s, __blocksize), 0xf4292244, 6)
		STEP(I, d, a, b, c, GET(7, ptr_s, __blocksize), 0x432aff97, 10)
		STEP(I, c, d, a, b, GET(14, ptr_s, __blocksize), 0xab9423a7, 15)
		STEP(I, b, c, d, a, GET(5, ptr_s, __blocksize), 0xfc93a039, 21)
		STEP(I, a, b, c, d, GET(12, ptr_s, __blocksize), 0x655b59c3, 6)
		STEP(I, d, a, b, c, GET(3, ptr_s, __blocksize), 0x8f0ccc92, 10)
		STEP(I, c, d, a, b, GET(10, ptr_s, __blocksize), 0xffeff47d, 15)
		STEP(I, b, c, d, a, GET(1, ptr_s, __blocksize), 0x85845dd1, 21)
		STEP(I, a, b, c, d, GET(8, ptr_s, __blocksize), 0x6fa87e4f, 6)
		STEP(I, d, a, b, c, GET(15, ptr_s, __blocksize), 0xfe2ce6e0, 10)
		STEP(I, c, d, a, b, GET(6, ptr_s, __blocksize), 0xa3014314, 15)
		STEP(I, b, c, d, a, GET(13, ptr_s, __blocksize), 0x4e0811a1, 21)
		STEP(I, a, b, c, d, GET(4, ptr_s, __blocksize), 0xf7537e82, 6)
		STEP(I, d, a, b, c, GET(11, ptr_s, __blocksize), 0xbd3af235, 10)
		STEP(I, c, d, a, b, GET(2, ptr_s, __blocksize), 0x2ad7d2bb, 15)
		STEP(I, b, c, d, a, GET(9, ptr_s, __blocksize), 0xeb86d391, 21)

		a += saved_a;
		b += saved_b;
		c += saved_c;
		d += saved_d;

		ptr += 64*offset;
	} while (size -= 64);

	ctx->a = a;
	ctx->b = b;
	ctx->c = c;
	ctx->d = d;

	return ptr;
}

__device__ void *body_with_local(MD5_CTX *ctx, void *data, unsigned long size)
{
	unsigned int *ptr;
	MD5_u32plus a, b, c, d;
	MD5_u32plus saved_a, saved_b, saved_c, saved_d;

	ptr = (unsigned int *)data;

	a = ctx->a;
	b = ctx->b;
	c = ctx->c;
	d = ctx->d;

	do {
		saved_a = a;
		saved_b = b;
		saved_c = c;
		saved_d = d;

/* Round 1 */
		STEP(F, a, b, c, d, SET(0, ptr, 1), 0xd76aa478, 7)
		STEP(F, d, a, b, c, SET(1, ptr, 1), 0xe8c7b756, 12)
		STEP(F, c, d, a, b, SET(2, ptr, 1), 0x242070db, 17)
		STEP(F, b, c, d, a, SET(3, ptr, 1), 0xc1bdceee, 22)
		STEP(F, a, b, c, d, SET(4, ptr, 1), 0xf57c0faf, 7)
		STEP(F, d, a, b, c, SET(5, ptr, 1), 0x4787c62a, 12)
		STEP(F, c, d, a, b, SET(6, ptr, 1), 0xa8304613, 17)
		STEP(F, b, c, d, a, SET(7, ptr, 1), 0xfd469501, 22)
		STEP(F, a, b, c, d, SET(8, ptr, 1), 0x698098d8, 7)
		STEP(F, d, a, b, c, SET(9, ptr, 1), 0x8b44f7af, 12)
		STEP(F, c, d, a, b, SET(10, ptr, 1), 0xffff5bb1, 17)
		STEP(F, b, c, d, a, SET(11, ptr, 1), 0x895cd7be, 22)
		STEP(F, a, b, c, d, SET(12, ptr, 1), 0x6b901122, 7)
		STEP(F, d, a, b, c, SET(13, ptr, 1), 0xfd987193, 12)
		STEP(F, c, d, a, b, SET(14, ptr, 1), 0xa679438e, 17)
		STEP(F, b, c, d, a, SET(15, ptr, 1), 0x49b40821, 22)

/* Round 2 */
		STEP(G, a, b, c, d, GET(1, ptr, 1), 0xf61e2562, 5)
		STEP(G, d, a, b, c, GET(6, ptr, 1), 0xc040b340, 9)
		STEP(G, c, d, a, b, GET(11, ptr, 1), 0x265e5a51, 14)
		STEP(G, b, c, d, a, GET(0, ptr, 1), 0xe9b6c7aa, 20)
		STEP(G, a, b, c, d, GET(5, ptr, 1), 0xd62f105d, 5)
		STEP(G, d, a, b, c, GET(10, ptr, 1), 0x02441453, 9)
		STEP(G, c, d, a, b, GET(15, ptr, 1), 0xd8a1e681, 14)
		STEP(G, b, c, d, a, GET(4, ptr, 1), 0xe7d3fbc8, 20)
		STEP(G, a, b, c, d, GET(9, ptr, 1), 0x21e1cde6, 5)
		STEP(G, d, a, b, c, GET(14, ptr, 1), 0xc33707d6, 9)
		STEP(G, c, d, a, b, GET(3, ptr, 1), 0xf4d50d87, 14)
		STEP(G, b, c, d, a, GET(8, ptr, 1), 0x455a14ed, 20)
		STEP(G, a, b, c, d, GET(13, ptr, 1), 0xa9e3e905, 5)
		STEP(G, d, a, b, c, GET(2, ptr, 1), 0xfcefa3f8, 9)
		STEP(G, c, d, a, b, GET(7, ptr, 1), 0x676f02d9, 14)
		STEP(G, b, c, d, a, GET(12, ptr, 1), 0x8d2a4c8a, 20)

/* Round 3 */
		STEP(H, a, b, c, d, GET(5, ptr, 1), 0xfffa3942, 4)
		STEP(H, d, a, b, c, GET(8, ptr, 1), 0x8771f681, 11)
		STEP(H, c, d, a, b, GET(11, ptr, 1), 0x6d9d6122, 16)
		STEP(H, b, c, d, a, GET(14, ptr, 1), 0xfde5380c, 23)
		STEP(H, a, b, c, d, GET(1, ptr, 1), 0xa4beea44, 4)
		STEP(H, d, a, b, c, GET(4, ptr, 1), 0x4bdecfa9, 11)
		STEP(H, c, d, a, b, GET(7, ptr, 1), 0xf6bb4b60, 16)
		STEP(H, b, c, d, a, GET(10, ptr, 1), 0xbebfbc70, 23)
		STEP(H, a, b, c, d, GET(13, ptr, 1), 0x289b7ec6, 4)
		STEP(H, d, a, b, c, GET(0, ptr, 1), 0xeaa127fa, 11)
		STEP(H, c, d, a, b, GET(3, ptr, 1), 0xd4ef3085, 16)
		STEP(H, b, c, d, a, GET(6, ptr, 1), 0x04881d05, 23)
		STEP(H, a, b, c, d, GET(9, ptr, 1), 0xd9d4d039, 4)
		STEP(H, d, a, b, c, GET(12, ptr, 1), 0xe6db99e5, 11)
		STEP(H, c, d, a, b, GET(15, ptr, 1), 0x1fa27cf8, 16)
		STEP(H, b, c, d, a, GET(2, ptr, 1), 0xc4ac5665, 23)

/* Round 4 */
		STEP(I, a, b, c, d, GET(0, ptr, 1), 0xf4292244, 6)
		STEP(I, d, a, b, c, GET(7, ptr, 1), 0x432aff97, 10)
		STEP(I, c, d, a, b, GET(14, ptr, 1), 0xab9423a7, 15)
		STEP(I, b, c, d, a, GET(5, ptr, 1), 0xfc93a039, 21)
		STEP(I, a, b, c, d, GET(12, ptr, 1), 0x655b59c3, 6)
		STEP(I, d, a, b, c, GET(3, ptr, 1), 0x8f0ccc92, 10)
		STEP(I, c, d, a, b, GET(10, ptr, 1), 0xffeff47d, 15)
		STEP(I, b, c, d, a, GET(1, ptr, 1), 0x85845dd1, 21)
		STEP(I, a, b, c, d, GET(8, ptr, 1), 0x6fa87e4f, 6)
		STEP(I, d, a, b, c, GET(15, ptr, 1), 0xfe2ce6e0, 10)
		STEP(I, c, d, a, b, GET(6, ptr, 1), 0xa3014314, 15)
		STEP(I, b, c, d, a, GET(13, ptr, 1), 0x4e0811a1, 21)
		STEP(I, a, b, c, d, GET(4, ptr, 1), 0xf7537e82, 6)
		STEP(I, d, a, b, c, GET(11, ptr, 1), 0xbd3af235, 10)
		STEP(I, c, d, a, b, GET(2, ptr, 1), 0x2ad7d2bb, 15)
		STEP(I, b, c, d, a, GET(9, ptr, 1), 0xeb86d391, 21)

		a += saved_a;
		b += saved_b;
		c += saved_c;
		d += saved_d;

		ptr += 16;
	} while (size -= 64);

	ctx->a = a;
	ctx->b = b;
	ctx->c = c;
	ctx->d = d;

	return ptr;
}

__device__ void MD5_Init(MD5_CTX *ctx)
{
	ctx->a = 0x67452301;
	ctx->b = 0xefcdab89;
	ctx->c = 0x98badcfe;
	ctx->d = 0x10325476;

	ctx->lo = 0;
	ctx->hi = 0;
}

__device__ void MD5_Update(MD5_CTX *ctx, void *data, long offset, unsigned long size)
{
	MD5_u32plus saved_lo;
	unsigned long used, free;

	saved_lo = ctx->lo;
	if ((ctx->lo = (saved_lo + size) & 0x1fffffff) < saved_lo)
		ctx->hi++;
	ctx->hi += size >> 29;

	used = saved_lo & 0x3f;

	uint8_t *p1;
	uint8_t *p2;
	if (used) {
		free = 64 - used;

		if (size < free) {
			//memcpy(&ctx->buffer[used], data, size);
			p1 = (uint8_t *)&ctx->buffer[used];
			p2 = (uint8_t *)data;
			for(int i=0; i<size; i++)
				p1[i] = p2[i*offset];
			return;
		}

		//memcpy(&ctx->buffer[used], data, free);
		p1 = (uint8_t *)&ctx->buffer[used];
		p2 = (uint8_t *)data;
		for(int i=0; i<free; i++)
			p1[i] = p2[i*offset];
		data = (unsigned char *)data + free*offset;
		size -= free;
		body_with_local(ctx, ctx->buffer, 64);
	}

	//printf("%d: %u %u %u %u %ld\n", threadIdx.x, ctx->a, ctx->b, ctx->c, ctx->d, size);
	if (size >= 64) {
		data = body_with_global(ctx, data, offset, size & ~(unsigned long)0x3f);
		size &= 0x3f;
	}

	//memcpy(ctx->buffer, data, size);
	p1 = (uint8_t *)ctx->buffer;
	p2 = (uint8_t *)data;
	for(int i=0; i<size; i++)
		p1[i] = p2[i*offset];
}

__device__ void MD5_Final(unsigned char *result, MD5_CTX *ctx, int offset)
{
	unsigned long used, free;

	used = ctx->lo & 0x3f;

	ctx->buffer[used++] = 0x80;

	free = 64 - used;

	if (free < 8) {
		//memset(&ctx->buffer[used], 0, free);
		uint8_t *p = (uint8_t *)&ctx->buffer[used];
		for(int i=0; i<free; i++)
			p[i]=0;
		body_with_local(ctx, ctx->buffer, 64);
		used = 0;
		free = 64;
	}

	//memset(&ctx->buffer[used], 0, free - 8);
	uint8_t *p = (uint8_t *)&ctx->buffer[used];
	for(int i=0; i<free-8; i++)
		p[i]=0;
	ctx->lo <<= 3;
	ctx->buffer[56] = ctx->lo;
	ctx->buffer[57] = ctx->lo >> 8;
	ctx->buffer[58] = ctx->lo >> 16;
	ctx->buffer[59] = ctx->lo >> 24;
	ctx->buffer[60] = ctx->hi;
	ctx->buffer[61] = ctx->hi >> 8;
	ctx->buffer[62] = ctx->hi >> 16;
	ctx->buffer[63] = ctx->hi >> 24;

	body_with_local(ctx, ctx->buffer, 64);

	//printf("%d: %d %d %d %d \n", threadIdx.x, ctx->a, ctx->b, ctx->c, ctx->d);

	/*result[0] = ctx->a;
	result[1] = ctx->a >> 8;
	result[2] = ctx->a >> 16;
	result[3] = ctx->a >> 24;
	result[4] = ctx->b;
	result[5] = ctx->b >> 8;
	result[6] = ctx->b >> 16;
	result[7] = ctx->b >> 24;
	result[8] = ctx->c;
	result[9] = ctx->c >> 8;
	result[10] = ctx->c >> 16;
	result[11] = ctx->c >> 24;
	result[12] = ctx->d;
	result[13] = ctx->d >> 8;
	result[14] = ctx->d >> 16;
	result[15] = ctx->d >> 24;*/

	result[0*offset] = ctx->a;
	result[1*offset] = ctx->a >> 8;
	result[2*offset] = ctx->a >> 16;
	result[3*offset] = ctx->a >> 24;
	result[4*offset] = ctx->b;
	result[5*offset] = ctx->b >> 8;
	result[6*offset] = ctx->b >> 16;
	result[7*offset] = ctx->b >> 24;
	result[8*offset] = ctx->c;
	result[9*offset] = ctx->c >> 8;
	result[10*offset] = ctx->c >> 16;
	result[11*offset] = ctx->c >> 24;
	result[12*offset] = ctx->d;
	result[13*offset] = ctx->d >> 8;
	result[14*offset] = ctx->d >> 16;
	result[15*offset] = ctx->d >> 24;

	//memset(ctx, 0, sizeof(*ctx));
}










