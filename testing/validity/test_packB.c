#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int main(int argc, char *argv[])
{
    int nc, kc;
    if (argc != 3)
    {
        printf("Usage: %s kc nc\n", argv[0]);
        return 0;
    }
    kc = atoi(argv[1]);
    nc = atoi(argv[2]);

    float *b = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    float *B_pack = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    float *B_pack1 = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    for (int i = 0; i < nc * kc; i++)
    {
        b[i] = i;
    }

    // for (int i = 0; i < nc; i++)
    // {
    //     for (int j = 0; j < kc; j++)
    //     {
    //         printf("%2.0f;", b[i * kc + j]);
    //     }
    //     printf("\n");
    // }
    sgemm_pack_b_kc_nc(nc, kc, b, nc, B_pack);
    sgemm_pack_b_kc_nc_v0(nc, kc, b, nc, B_pack1);

    //验证正确性
    bool ifRight = true;
    //验证
    for (int j = 0; j < kc * nc; j++)
    {
        if (B_pack1[j] != B_pack[j])
        {
            ifRight = false;
        }
    }

    if (ifRight)
    {
        printf("pack success!\n");
    }
    else
    {
        printf("pack fails ):\n");
    }
    __aligned_free(b);
    __aligned_free(B_pack);
    __aligned_free(B_pack1);
    return 0;
}