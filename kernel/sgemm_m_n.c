#include "config.h"

// C row major, A row major, B row major
void sgemm_m_n(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    float *C)
{
    int mc, nc, kc;
    getOptimizeMCNCKC(m, n, k, &mc, &nc, &kc);
    // alloc A
    float *A_pack = (float *)__aligned_malloc(mc * kc * sizeof(float), PIDAN_PAGE_SIZE);
    // alloc B
    float *B_pack = (float *)__aligned_malloc(nc * kc * sizeof(float), PIDAN_PAGE_SIZE);

    int offset_a = 0, offset_b = 0;
    for (int i = 0; i < m; i += mc)
    {
        for (int j = 0; j < n; j += nc)
        {
        }
    }
}