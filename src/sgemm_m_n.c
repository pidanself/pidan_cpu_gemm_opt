#include "config.h"

// void printCol(const float *a, int m, int n, int lda)
// {

//     for (int j = 0; j < n; j++)
//     {
//         for (int i = 0; i < m; i++)
//         {
//             printf("%f,", a[i + j * lda]);
//         }
//         printf("\n");
//     }
//     printf("---------------\n");
// }

// void printRow(const float *a, int m, int n, int lda)
// {
//     for (int i = 0; i < m; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             printf("%f,", a[i * lda + j]);
//         }
//         printf("\n");
//     }
//     printf("---------------\n");
// }

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

    for (int mm = 0; mm < m; mm += mc)
    {
        mc_size = min(m - mm, mc);
        for (int kk = 0; kk < k; kk += kc)
        {
            kc_size = min(k - kk, kc);
            // printRow(A, m, n, n);
            sgemm_pack_a_mc_kc(mc_size, kc_size, A + kk + mm * k, k, A_pack);
            // printCol(A_pack, m, n, m);
            for (int nn = 0; nn < n; nn += nc)
            {
                nc_size = min(n - nn, nc);
                // printf("mc %d;nc %d;kc %d\n", mm, nn, kk);
                // printRow(B, k, n, n);
                sgemm_pack_b_kc_nc(nc_size, kc_size, B + kk * n + nn, n, B_pack);
                // printRow(B_pack, k, n, n);
                sgemm_mc_nc(mc_size, nc_size, kc_size, A_pack, B_pack, C + mm * n + nn, n);
            }
        }
    }
    __aligned_free(A_pack);
    __aligned_free(B_pack);
}