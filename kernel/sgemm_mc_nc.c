#include "config.h"

// C row major, packA col major, packB row major
void sgemm_mc_nc(
    int mc,
    int nc,
    int kc,
    const float *packA,
    const float *packB,
    float *C,
    int ldc)
{
    int mr = PIDAN_MR;
    int nr = PIDAN_NR;
    int offset_a = 0, offset_b = 0;
    for (int i = 0; i < mc; i += mr)
    {
        for (int j = 0; j < nc; j += nr)
        {
            //理论上来讲，会有mc和nc不是mr和nr倍数的情况，暂不考虑
            sgemm_kernel_6x16(packA + offset_a, packB + offset_b, C + i * ldc + j, mr, nr, kc, ldc);
            offset_b += kc * nr;
        }
        offset_a += mr * kc;
        offset_b = 0;
    }
}