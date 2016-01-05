#ifndef H_MD5_MELEM_H
 
#define H_MD5_MELEM_H
 
/* Gestion des erreurs */
 
#define MD5_SUCCEEDED(n) ((n) == 0)
 
enum e_md5_errors {
    MD5_SUCCESS, MD5_INTERNAL_ERROR, MD5_INVALID_ARG, MD5_NO_MEM
};
 
/* Valeurs initiales des registres A, B, C et D */
 
#define A0 0x67452301
#define B0 0xefcdab89
#define C0 0x98badcfe
#define D0 0x10325476
 
/* Definition des macros F, G, H et I */
 
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
 
/* Implementation de l'operateur <<< (rotate left) */
 
#define ROL(x, n) (((x) << (n)) | ((x) >> (32-(n))))
 
/* Definition des macros FXT, GXT, HXT et IXT et des constantes Snn */
 
#define FXT(a, b, c, d, x, s, t) ((a) = (b) + ROL(((a) + F((b), (c), (d)) + (x) + (t)), (s)))
#define GXT(a, b, c, d, x, s, t) ((a) = (b) + ROL(((a) + G((b), (c), (d)) + (x) + (t)), (s)))
#define HXT(a, b, c, d, x, s, t) ((a) = (b) + ROL(((a) + H((b), (c), (d)) + (x) + (t)), (s)))
#define IXT(a, b, c, d, x, s, t) ((a) = (b) + ROL(((a) + I((b), (c), (d)) + (x) + (t)), (s)))
 
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21
 
/* Declaration des fonctions */
 
int  md5_init(void);
// int  md5(char * sum, const char * m, int bytes);
void md5_format(char * output, const char * sum);
void md5_format_ex(char * output, const char * buf, int bytes);
// __device__ int  md5_get_err_number(void);
// __device__ void md5_set_err_number(int n);
 
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
static int _md5_errno = MD5_SUCCESS;
static unsigned hostT[64];
__constant__ unsigned T[64];
 
__device__ static int md5_comput(char * sum, const char * buf, int bytes);
 
int md5_init()
{
    int ret = 0;
    FILE * f = fopen("t.data", "rb");
 
    if (f == NULL)
    {
        fprintf(stderr, "Could not find data file\n");
        ret = -1;
    }
    else
    {
        size_t n = fread(hostT, sizeof(hostT[0]), sizeof(hostT) / sizeof(hostT[0]), f);
 
        if (n != 64)
        {
            ret = -1;
        }
 
        fclose(f);
    }

    cudaMemcpyToSymbol(T, hostT, 64 * sizeof(unsigned));
 
    return ret;
}
 
__device__ int md5(char * sum, const char * m, int bytes) /* Calcule le MD5 d'un message */
{
    int ret = 0;
 
    /* Valider les arguments */
 
    if (sum == NULL || m == NULL || bytes < 0)
    {
        /* Arguments invalides */
 
        ret = -1;
        // md5_set_err_number(MD5_INVALID_ARG);
    }
    else
    {
        /* Calculer la memoire necessaire pour la phase initiale */
 
        int N, mod, pad = 0;
        char * M;
 
        N = bytes + 1;
        mod = N % 64;
 
        if (mod != 56)
            pad = (N < 56) ? 56 - mod : 56 + 64 - mod;
 
        M = (char*)malloc(bytes + 1 + pad + 8);
 
        if (M == NULL)
        {
            /* Le systeme n'a pas assez de memoire */
 
            ret = -1;
            // md5_set_err_number(MD5_NO_MEM);
        }
        else
        {
            /* Normaliser le message */
 
            unsigned long int n = bytes * 8; /* Longueur du message en bits */
            int N = bytes;
 
            memcpy(M, m, bytes);
 
            M[N] = '\x80'; /* Ajouter "10000000" */
            N++;
 
            if (pad != 0)
            {
                memset(M + N, (unsigned char)0, pad); /* Completer par des "0" */
                N += pad;
            }
 
            memcpy(M + bytes + 1 + pad, &n, sizeof(n)); /* Ajouter "n" */
            N += 8;
 
            /* Calculer le MD5 */
 
            ret = md5_comput(sum, M, N);
 
            free(M);
        }
    }
 
    return ret;
}
 
__device__ int md5_comput(char * sum, const char * buf, int bytes) /* Calcule le MD5 d'un message normalise */
{
    int ret = 0;
    const unsigned * M = (const unsigned *)buf;
    int i, N = bytes / 4;
    unsigned r[4] = {A0, B0, C0, D0}; /* Registres A, B, C et D */
 
    for(i = 0; i < N / 16; i++)
    {
        const unsigned * x = (M + (16 * i));
        int j;
        unsigned s[4];
 
        for(j = 0; j < 4; j++)
            s[j] = r[j];
 
        j = 0;
 
        FXT(r[0], r[1], r[2], r[3], x[ 0], S11, T[j]); j++;
        FXT(r[3], r[0], r[1], r[2], x[ 1], S12, T[j]); j++;
        FXT(r[2], r[3], r[0], r[1], x[ 2], S13, T[j]); j++;
        FXT(r[1], r[2], r[3], r[0], x[ 3], S14, T[j]); j++;
        FXT(r[0], r[1], r[2], r[3], x[ 4], S11, T[j]); j++;
        FXT(r[3], r[0], r[1], r[2], x[ 5], S12, T[j]); j++;
        FXT(r[2], r[3], r[0], r[1], x[ 6], S13, T[j]); j++;
        FXT(r[1], r[2], r[3], r[0], x[ 7], S14, T[j]); j++;
        FXT(r[0], r[1], r[2], r[3], x[ 8], S11, T[j]); j++;
        FXT(r[3], r[0], r[1], r[2], x[ 9], S12, T[j]); j++;
        FXT(r[2], r[3], r[0], r[1], x[10], S13, T[j]); j++;
        FXT(r[1], r[2], r[3], r[0], x[11], S14, T[j]); j++;
        FXT(r[0], r[1], r[2], r[3], x[12], S11, T[j]); j++;
        FXT(r[3], r[0], r[1], r[2], x[13], S12, T[j]); j++;
        FXT(r[2], r[3], r[0], r[1], x[14], S13, T[j]); j++;
        FXT(r[1], r[2], r[3], r[0], x[15], S14, T[j]); j++;
 
        GXT(r[0], r[1], r[2], r[3], x[ 1], S21, T[j]); j++;
        GXT(r[3], r[0], r[1], r[2], x[ 6], S22, T[j]); j++;
        GXT(r[2], r[3], r[0], r[1], x[11], S23, T[j]); j++;
        GXT(r[1], r[2], r[3], r[0], x[ 0], S24, T[j]); j++;
        GXT(r[0], r[1], r[2], r[3], x[ 5], S21, T[j]); j++;
        GXT(r[3], r[0], r[1], r[2], x[10], S22, T[j]); j++;
        GXT(r[2], r[3], r[0], r[1], x[15], S23, T[j]); j++;
        GXT(r[1], r[2], r[3], r[0], x[ 4], S24, T[j]); j++;
        GXT(r[0], r[1], r[2], r[3], x[ 9], S21, T[j]); j++;
        GXT(r[3], r[0], r[1], r[2], x[14], S22, T[j]); j++;
        GXT(r[2], r[3], r[0], r[1], x[ 3], S23, T[j]); j++;
        GXT(r[1], r[2], r[3], r[0], x[ 8], S24, T[j]); j++;
        GXT(r[0], r[1], r[2], r[3], x[13], S21, T[j]); j++;
        GXT(r[3], r[0], r[1], r[2], x[ 2], S22, T[j]); j++;
        GXT(r[2], r[3], r[0], r[1], x[ 7], S23, T[j]); j++;
        GXT(r[1], r[2], r[3], r[0], x[12], S24, T[j]); j++;
 
        HXT(r[0], r[1], r[2], r[3], x[ 5], S31, T[j]); j++;
        HXT(r[3], r[0], r[1], r[2], x[ 8], S32, T[j]); j++;
        HXT(r[2], r[3], r[0], r[1], x[11], S33, T[j]); j++;
        HXT(r[1], r[2], r[3], r[0], x[14], S34, T[j]); j++;
        HXT(r[0], r[1], r[2], r[3], x[ 1], S31, T[j]); j++;
        HXT(r[3], r[0], r[1], r[2], x[ 4], S32, T[j]); j++;
        HXT(r[2], r[3], r[0], r[1], x[ 7], S33, T[j]); j++;
        HXT(r[1], r[2], r[3], r[0], x[10], S34, T[j]); j++;
        HXT(r[0], r[1], r[2], r[3], x[13], S31, T[j]); j++;
        HXT(r[3], r[0], r[1], r[2], x[ 0], S32, T[j]); j++;
        HXT(r[2], r[3], r[0], r[1], x[ 3], S33, T[j]); j++;
        HXT(r[1], r[2], r[3], r[0], x[ 6], S34, T[j]); j++;
        HXT(r[0], r[1], r[2], r[3], x[ 9], S31, T[j]); j++;
        HXT(r[3], r[0], r[1], r[2], x[12], S32, T[j]); j++;
        HXT(r[2], r[3], r[0], r[1], x[15], S33, T[j]); j++;
        HXT(r[1], r[2], r[3], r[0], x[ 2], S34, T[j]); j++;
 
        IXT(r[0], r[1], r[2], r[3], x[ 0], S41, T[j]); j++;
        IXT(r[3], r[0], r[1], r[2], x[ 7], S42, T[j]); j++;
        IXT(r[2], r[3], r[0], r[1], x[14], S43, T[j]); j++;
        IXT(r[1], r[2], r[3], r[0], x[ 5], S44, T[j]); j++;
        IXT(r[0], r[1], r[2], r[3], x[12], S41, T[j]); j++;
        IXT(r[3], r[0], r[1], r[2], x[ 3], S42, T[j]); j++;
        IXT(r[2], r[3], r[0], r[1], x[10], S43, T[j]); j++;
        IXT(r[1], r[2], r[3], r[0], x[ 1], S44, T[j]); j++;
        IXT(r[0], r[1], r[2], r[3], x[ 8], S41, T[j]); j++;
        IXT(r[3], r[0], r[1], r[2], x[15], S42, T[j]); j++;
        IXT(r[2], r[3], r[0], r[1], x[ 6], S43, T[j]); j++;
        IXT(r[1], r[2], r[3], r[0], x[13], S44, T[j]); j++;
        IXT(r[0], r[1], r[2], r[3], x[ 4], S41, T[j]); j++;
        IXT(r[3], r[0], r[1], r[2], x[11], S42, T[j]); j++;
        IXT(r[2], r[3], r[0], r[1], x[ 2], S43, T[j]); j++;
        IXT(r[1], r[2], r[3], r[0], x[ 9], S44, T[j]); j++;
 
        for(j = 0; j < 4; j++)
            r[j] += s[j];
    }
 
    memcpy(sum, r, 16);
 
    return ret;
}
 
void md5_format(char * output, const char * sum)
{
    md5_format_ex(output, sum, 16);
}
 
void md5_format_ex(char * output, const char * m, int bytes)
{
    int i;
 
    for(i = 0; i < bytes; i++)
    {
        sprintf(output, "%02x", (unsigned char)m[i]);
        output += 2;
    }
}
 
// __device__ int md5_get_err_number()
// {
//     return _md5_errno;
// }
 
// __device__ void md5_set_err_number(int n)
// {
//     _md5_errno = n;
// }