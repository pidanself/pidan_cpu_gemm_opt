#include "config.h"

void sgemm_pack_a_mc_kc(int mc, int kc, const float *A, int lda, float *A_pack)
{
    int mr = PIDAN_MR;
    // v0 no optimize
    for (int i = 0; i < mc; i += mr)
    {
        for (int mm = 0; mm < mr; mm++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                A_pack[i * kc + mm + kk * mr] = A[(i + mm) * lda + kk];
            }
        }
    }
    // v1 intrinsics
    // v2 asm
}
void sgemm_pack_b_kc_nc(int nc, int kc, const float *B, int ldb, float *B_pack)
{
    int nr = PIDAN_NR;
    // v0 no optimize
    for (int i = 0; i < nc; i += nr)
    {
        for (int kk = 0; kk < kc; kk++)
        {
            for (int nn = 0; nn < nc; nn++)
            {
                B_pack[i * kc + nn + kk * nr] = B[i + nn + kk * ldb];
            }
        }
    }
    // v1 intrinsics
    // v2 asm
}