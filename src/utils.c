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
    // TODO: tune to get appropriate mc nc and kc
    if (m == 4128 && n == 4128 && k == 4128)
    {
        *mc = 4128;
        *nc = 448;
        *kc = 352;
    }
    if (m == 48 && n == 48 && k == 48)
    {
        *mc = 48;
        *nc = 48;
        *kc = 48;
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