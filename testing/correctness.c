#include <stdio.h>
#include <stdlib.h>
#include "pidanGemm.h"
#include <cblas.h>
#include <math.h>
#include <stdbool.h>

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

    float *a = (float *)malloc(2 * m * k * sizeof(float));
    float *b = (float *)malloc(2 * n * k * sizeof(float));
    float *c = (float *)malloc(2 * m * n * sizeof(float));

    for (int i = 0; i < m * k; i++)
    {
        a[i] = (rand() % 100) / 10.0;
        a[i + m * k] = a[i];
    }
    for (int i = 0; i < k * n; i++)
    {
        b[i] = (rand() % 100) / 10.0;
        b[i + n * k] = b[i];
    }
    for (int i = 0; i < m * n; i++)
    {
        c[i] = 0;
        c[i + m * n] = c[i];
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
            if (fabs(c[i * n + j] - c[i * n + j + m * n]) > 0.0001f)
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

    free(a);
    free(b);
    free(c);

    return 0;
}