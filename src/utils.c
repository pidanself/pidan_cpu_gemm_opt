#include <stddef.h>
#include <stdlib.h>

void *__aligned_malloc(size_t required_bytes, size_t alignment)
{
    if (alignment == 0 || (alignment & (alignment - 1))) // check pow of 2
        return NULL;
    void *p1;  // original block
    void **p2; // aligned block
    int offset = alignment - 1 + sizeof(void *);
    if ((p1 = (void *)malloc(required_bytes + offset)) == NULL)
    {
        return NULL;
    }
    p2 = (void **)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void __aligned_free(void *p)
{
    free(((void **)p)[-1]);
}

void getOptimizeMCNCKC(int m, int n, int k, int *mc, int *nc, int *kc)
{
    // get 84% cpu peak performance
    if (m == 4128 && n == 4128 && k == 4128)
    {
        *mc = 1872; // 1872
        *nc = 240;  // 240
        *kc = 160;  // 160
    }

    if (m == 1152 && n == 1152 && k == 1152)
    {
        *mc = 1152; // 1872
        *nc = 224;  // 240
        *kc = 160;  // 160
    }

    if (m == 48 && n == 48 && k == 48)
    {
        *mc = 48;
        *nc = 16;
        *kc = 48;
    }
    if (m == 192 && n == 192 && k == 192)
    {
        *mc = 192;
        *nc = 144;
        *kc = 140;
    }
    if (m == 1152 && n == 1152 && k == 1152)
    {
        *mc = 1152;
        *nc = 240;
        *kc = 160;
    }
}

int min(int a, int b)
{
    return a < b ? a : b;
}

double minf(double a, double b)
{
    return a < b ? a : b;
}