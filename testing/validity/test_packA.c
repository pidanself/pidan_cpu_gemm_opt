#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int main(int argc, char *argv[])
{
    int mc, kc;
    if (argc != 3)
    {
        printf("Usage: %s mc kc\n", argv[0]);
        return 0;
    }
    mc = atoi(argv[1]);
    kc = atoi(argv[2]);

    float *a = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    float *A_pack = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    float *A_pack1 = (float *)__aligned_malloc(8192 * sizeof(float), 32);
    for (int i = 0; i < mc * kc; i++)
    {
        a[i] = i;
    }

    // for (int i = 0; i < mc; i++)
    // {
    //     for (int j = 0; j < kc; j++)
    //     {
    //         printf("%2.0f;", a[i * kc + j]);
    //     }
    //     printf("\n");
    // }
    sgemm_pack_a_mc_kc(mc, kc, a, kc, A_pack);

    //验证正确性
    float *tempA_pack = A_pack;
    int m_itr = mc / 6;
    int m_rem = mc % 6;
    int index = 0;
    for (int k = 0; k < m_itr; k++)
    {
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < kc; j++)
            {
                // printf("%2.0f;", tempA_pack[i + j * 6]);
                A_pack1[index++] = tempA_pack[i + j * 6];
            }
            // printf("\n");
        }
        tempA_pack += 6 * kc;
    }

    for (int k = 0; k < m_rem; k++)
    {
        for (int j = 0; j < kc; j++)
        {
            // printf("%2.0f;", tempA_pack[k + j * m_rem]);
            A_pack1[index++] = tempA_pack[k + j * m_rem];
        }
        // printf("\n");
    }

    bool ifRight = true;
    //验证
    for (int i = 0; i < mc; i++)
    {
        for (int j = 0; j < kc; j++)
        {
            if (a[i * kc + j] != A_pack1[i * kc + j])
            {
                ifRight = false;
            }
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
    __aligned_free(a);
    __aligned_free(A_pack);
    __aligned_free(A_pack1);
    return 0;
}