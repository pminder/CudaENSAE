#ifndef H_MD5_MELEM_H
 
#define H_MD5_MELEM_H
 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define rotl(x,n,w) ((x << n) | (x >> (w - n)))
#define rotl32(x,n) rotl(x,n,32) 

#define uint unsigned int
#define MAGIC_A 0x67452301
#define MAGIC_B 0xefcdab89
#define MAGIC_C 0x98badcfe
#define MAGIC_D 0x10325476

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (ac); \
   (a) = rotl32 ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (ac); \
   (a) = rotl32 ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (ac); \
   (a) = rotl32 ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (ac); \
   (a) = rotl32 ((a), (s)); \
   (a) += (b); \
  }


//Compute md5 hash of string m. Stores it in sum. Uses M as temporary buffer. m
//must be of size bytes (number of bytes)
__device__ void md5(char * sum, char * m, char * M, int bytes);
//Compute md5 hash of a normalized message block (stores it in digest). Many
//optimizations have been made, knowing that we do not intend to brute force
//passwords longer than 10 chars :)
__device__ void md5_comput(unsigned int * block, unsigned int * digest);


__device__ void md5(char * sum, char * m, char * M, int bytes)
{
    /* Normaliser le message */
    for (int i = 0; i < bytes; ++i) {
        M[i] = m[i];
    }
    M[bytes] = '\x80'; /* Ajouter "10000000" */
    ((unsigned int*)M)[14] = 8*bytes;

    /* Calculer le MD5 */
    md5_comput((unsigned int *)M, (unsigned int *)sum);
}

__device__ void md5_comput(unsigned int * block, unsigned int * digest)
{
  digest[0] = MAGIC_A;
  digest[1] = MAGIC_B;
  digest[2] = MAGIC_C;
  digest[3] = MAGIC_D;

  #define S11 7
  #define S12 12
  #define S13 17
  #define S14 22

  FF ( digest[0], digest[1], digest[2], digest[3], block[ 0], S11, 0xd76aa478);
  FF ( digest[3], digest[0], digest[1], digest[2], block[ 1], S12, 0xe8c7b756);
  FF ( digest[2], digest[3], digest[0], digest[1], block[ 2], S13, 0x242070db);
  FF ( digest[1], digest[2], digest[3], digest[0], block[ 3], S14, 0xc1bdceee);
  FF ( digest[0], digest[1], digest[2], digest[3], block[ 4], S11, 0xf57c0faf);
  FF ( digest[3], digest[0], digest[1], digest[2], block[ 5], S12, 0x4787c62a);
  FF ( digest[2], digest[3], digest[0], digest[1], block[ 6], S13, 0xa8304613);
  FF ( digest[1], digest[2], digest[3], digest[0], block[ 7], S14, 0xfd469501);
  FF ( digest[0], digest[1], digest[2], digest[3], block[ 8], S11, 0x698098d8);
  FF ( digest[3], digest[0], digest[1], digest[2], block[ 9], S12, 0x8b44f7af);
  FF ( digest[2], digest[3], digest[0], digest[1], block[10], S13, 0xffff5bb1);
  FF ( digest[1], digest[2], digest[3], digest[0], block[11], S14, 0x895cd7be);
  FF ( digest[0], digest[1], digest[2], digest[3], block[12], S11, 0x6b901122);
  FF ( digest[3], digest[0], digest[1], digest[2], block[13], S12, 0xfd987193);
  FF ( digest[2], digest[3], digest[0], digest[1], block[14], S13, 0xa679438e);
  FF ( digest[1], digest[2], digest[3], digest[0], block[15], S14, 0x49b40821);

  #define S21 5
  #define S22 9
  #define S23 14
  #define S24 20

  GG ( digest[0], digest[1], digest[2], digest[3], block[ 1], S21, 0xf61e2562);
  GG ( digest[3], digest[0], digest[1], digest[2], block[ 6], S22, 0xc040b340);
  GG ( digest[2], digest[3], digest[0], digest[1], block[11], S23, 0x265e5a51);
  GG ( digest[1], digest[2], digest[3], digest[0], block[ 0], S24, 0xe9b6c7aa);
  GG ( digest[0], digest[1], digest[2], digest[3], block[ 5], S21, 0xd62f105d);
  GG ( digest[3], digest[0], digest[1], digest[2], block[10], S22, 0x02441453);
  GG ( digest[2], digest[3], digest[0], digest[1], block[15], S23, 0xd8a1e681);
  GG ( digest[1], digest[2], digest[3], digest[0], block[ 4], S24, 0xe7d3fbc8);
  GG ( digest[0], digest[1], digest[2], digest[3], block[ 9], S21, 0x21e1cde6);
  GG ( digest[3], digest[0], digest[1], digest[2], block[14], S22, 0xc33707d6);
  GG ( digest[2], digest[3], digest[0], digest[1], block[ 3], S23, 0xf4d50d87);
  GG ( digest[1], digest[2], digest[3], digest[0], block[ 8], S24, 0x455a14ed);
  GG ( digest[0], digest[1], digest[2], digest[3], block[13], S21, 0xa9e3e905);
  GG ( digest[3], digest[0], digest[1], digest[2], block[ 2], S22, 0xfcefa3f8);
  GG ( digest[2], digest[3], digest[0], digest[1], block[ 7], S23, 0x676f02d9);
  GG ( digest[1], digest[2], digest[3], digest[0], block[12], S24, 0x8d2a4c8a);

  #define S31 4
  #define S32 11
  #define S33 16
  #define S34 23

  HH ( digest[0], digest[1], digest[2], digest[3], block[ 5], S31, 0xfffa3942);
  HH ( digest[3], digest[0], digest[1], digest[2], block[ 8], S32, 0x8771f681);
  HH ( digest[2], digest[3], digest[0], digest[1], block[11], S33, 0x6d9d6122);
  HH ( digest[1], digest[2], digest[3], digest[0], block[14], S34, 0xfde5380c);
  HH ( digest[0], digest[1], digest[2], digest[3], block[ 1], S31, 0xa4beea44);
  HH ( digest[3], digest[0], digest[1], digest[2], block[ 4], S32, 0x4bdecfa9);
  HH ( digest[2], digest[3], digest[0], digest[1], block[ 7], S33, 0xf6bb4b60);
  HH ( digest[1], digest[2], digest[3], digest[0], block[10], S34, 0xbebfbc70);
  HH ( digest[0], digest[1], digest[2], digest[3], block[13], S31, 0x289b7ec6);
  HH ( digest[3], digest[0], digest[1], digest[2], block[ 0], S32, 0xeaa127fa);
  HH ( digest[2], digest[3], digest[0], digest[1], block[ 3], S33, 0xd4ef3085);
  HH ( digest[1], digest[2], digest[3], digest[0], block[ 6], S34, 0x04881d05);
  HH ( digest[0], digest[1], digest[2], digest[3], block[ 9], S31, 0xd9d4d039);
  HH ( digest[3], digest[0], digest[1], digest[2], block[12], S32, 0xe6db99e5);
  HH ( digest[2], digest[3], digest[0], digest[1], block[15], S33, 0x1fa27cf8);
  HH ( digest[1], digest[2], digest[3], digest[0], block[ 2], S34, 0xc4ac5665);

  #define S41 6
  #define S42 10
  #define S43 15
  #define S44 21

  II ( digest[0], digest[1], digest[2], digest[3], block[ 0], S41, 0xf4292244);
  II ( digest[3], digest[0], digest[1], digest[2], block[ 7], S42, 0x432aff97);
  II ( digest[2], digest[3], digest[0], digest[1], block[14], S43, 0xab9423a7);
  II ( digest[1], digest[2], digest[3], digest[0], block[ 5], S44, 0xfc93a039);
  II ( digest[0], digest[1], digest[2], digest[3], block[12], S41, 0x655b59c3);
  II ( digest[3], digest[0], digest[1], digest[2], block[ 3], S42, 0x8f0ccc92);
  II ( digest[2], digest[3], digest[0], digest[1], block[10], S43, 0xffeff47d);
  II ( digest[1], digest[2], digest[3], digest[0], block[ 1], S44, 0x85845dd1);
  II ( digest[0], digest[1], digest[2], digest[3], block[ 8], S41, 0x6fa87e4f);
  II ( digest[3], digest[0], digest[1], digest[2], block[15], S42, 0xfe2ce6e0);
  II ( digest[2], digest[3], digest[0], digest[1], block[ 6], S43, 0xa3014314);
  II ( digest[1], digest[2], digest[3], digest[0], block[13], S44, 0x4e0811a1);
  II ( digest[0], digest[1], digest[2], digest[3], block[ 4], S41, 0xf7537e82);
  II ( digest[3], digest[0], digest[1], digest[2], block[11], S42, 0xbd3af235);
  II ( digest[2], digest[3], digest[0], digest[1], block[ 2], S43, 0x2ad7d2bb);
  II ( digest[1], digest[2], digest[3], digest[0], block[ 9], S44, 0xeb86d391);

  digest[0] += MAGIC_A;
  digest[1] += MAGIC_B;
  digest[2] += MAGIC_C;
  digest[3] += MAGIC_D;
}

#endif
