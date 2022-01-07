#include "config.h"

int min(int a, int b)
{
    return a < b ? a : b;
}

// C row major, A row major, B row major
void sgemm_m_n(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    float *C)
{
    int mc, nc, kc, mc_size, nc_size, kc_size;
    getOptimizeMCNCKC(m, n, k, &mc, &nc, &kc);

    // alloc A
    float *A_pack = (float *)__aligned_malloc(mc * kc * sizeof(float), PIDAN_PAGE_SIZE);
    // alloc B
    float *B_pack = (float *)__aligned_malloc(nc * kc * sizeof(float), PIDAN_PAGE_SIZE);
    int offset_a = 0, offset_b = 0;
    for (int mm = 0; mm < m; mm += mc)
    {
        mc_size = min(m - mm, mc);
        sgemm_pack_a_mc_kc(mc, nc, kc, A, k, A_pack);
        for (int kk = 0; kk < k; kk += kc)
        {
            kc_size = min(k - kk, kc);
            sgemm_pack_b_kc_nc(mc, nc, kc, B, n, B_pack);
            for (int nn = 0; nn < n; nn += nc)
            {
                nc_size = min(n - nn, nc);
                sgemm_mc_nc(mc_size, nc_size, kc_size, A_pack, B_pack, C + mm * n + nn, n);
            }
        }
    }
    __aligned_free(A_pack);
    __aligned_free(B_pack);
}