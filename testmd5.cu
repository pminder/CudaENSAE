#include <stdio.h>
#include <string.h>
#include "md5.h"

__global__ void kernel(char * devMd5sum, char * devMessage, size_t size)
{
    md5(devMd5sum, devMessage, size);
}
 
int main()
{
    md5_init();
    printf("MD5 test suite :\n");

    const char * t = "message digest";
    char mdsum[16], mdoutput[32 + 1];
    char * devMd5sum, *devMessage;

    cudaMalloc((void**)&devMd5sum, 16 * sizeof(char));
    cudaMalloc((void**)&devMessage, 15 * sizeof(char));
    cudaMemcpy(devMessage, t, 15 * sizeof(char), cudaMemcpyHostToDevice);

    kernel<<<1,1>>>(devMd5sum, devMessage, 14);

    cudaMemcpy(mdsum, devMd5sum, 16 * sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("md5(\"%s\") = ", t);

    md5_format(mdoutput, mdsum);
    printf("%s\n", mdoutput);
 
    return 0;
}