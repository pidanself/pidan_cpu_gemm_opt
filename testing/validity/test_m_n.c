#include <stdio.h>
#include <stdlib.h>
#include "pidanGemm.h"
#include <cblas.h>
#include <math.h>
#include <stdbool.h>
#include "config.h"

int main(int argc, char *argv[])
{
    int m, k, n;
    if (argc != 2)
    {
        printf("Usage: %s m\n", argv[0]);
        return 0;
    }
    m = atoi(argv[1]);
    k = m;
    n = m;

    float *a = (float *)__aligned_malloc(2 * m * k * sizeof(float), 32);
    float *b = (float *)__aligned_malloc(2 * n * k * sizeof(float), 32);
    float *c = (float *)__aligned_malloc(2 * m * n * sizeof(float), 32);

    for (int i = 0; i < m * k; i++)
    {
        a[i] = (rand() % 10) / 10.0;
        // a[i] = (rand() % 10);
        a[i + m * k] = a[i];
    }
    for (int i = 0; i < k * n; i++)
    {
        b[i] = (rand() % 10) / 10.0;
        // b[i] = (rand() % 10);
        b[i + n * k] = b[i];
    }
    for (int i = 0; i < 2 * m * n; i++)
    {
        c[i] = 0;
    }

    openblas_set_num_threads(1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a + m * k, k, b + n * k, n, 0, c + m * n, n);

    sgemm_m_n(m, n, k, a, b, c);
    // not precise
    bool right = true;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(c[i * n + j] - c[i * n + j + m * n]) > 0.01f)
            {
                right = false;
                printf("%dth row, %dth col; my %f; openblas %f; diff %f\n", i, j, c[i * n + j], c[i * n + j + m * n], fabs(c[i * n + j] - c[i * n + j + m * n]));
            }
        }
    }
    if (right)
    {
        printf("computing results is right\n");
    }
    else
    {
        printf("wrong!!!!\n");
    }

    __aligned_free(a);
    __aligned_free(b);
    __aligned_free(c);

    return 0;
}