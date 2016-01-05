#include <stdio.h>
#include <math.h>
 
int main()
{
    FILE * f = fopen("t.data", "wb");
 
    if (f == NULL)
        perror("t.data");
    else
    {
        unsigned int i, t[64];
        unsigned long int Tm = 0x100000000;
 
        for(i = 0; i < 64; i++)
            t[i] = (unsigned int)(Tm * fabs(sin(i + 1)));
 
        fwrite(t, sizeof(t[0]), sizeof(t) / sizeof(t[0]), f);
 
        fclose(f);
    }
 
    return 0;
}