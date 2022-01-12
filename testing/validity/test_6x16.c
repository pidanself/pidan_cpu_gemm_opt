#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include "pidanGemm.h"
#include "config.h"
#include <math.h>
#include <stdbool.h>

extern void sgemm_kernel_6x16(const float *A, const float *B, float *C, int m, int n, int k, int ldc);

static double
get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char *argv[])
{
    int m = 6, k, n = 16;
    if (argc != 2)
    {
        printf("Usage: %s k\n", argv[0]);
        return 0;
    }
    k = atoi(argv[1]);

    printf("all need %ld Bytes\n", 4L * ((m * k) + m * n + k * n));

    long comp = 2L * m * k * n;
    int loop_time = (int)(2e11 / comp);

    struct timespec start, end;
    double t, gflops;
    // m=4,k=16,n=8;
    float *a = (float *)__aligned_malloc(2 * m * k * sizeof(float), 32);
    float *b = (float *)__aligned_malloc(2 * n * k * sizeof(float), 32);
    float *c = (float *)__aligned_malloc(2 * m * n * sizeof(float), 32);

    for (int i = 0; i < m * k; i++)
    {
        a[i] = (rand() % 10) / 10.0;
        // a[i] = i % 10;
        a[i + m * k] = a[i];
    }
    for (int i = 0; i < k * n; i++)
    {
        b[i] = (rand() % 10) / 10.0;
        // b[i] = i % 10;
        b[i + n * k] = b[i];
    }
    for (int i = 0; i < 2 * m * n; i++)
    {
        c[i] = 0;
    }

    // check input
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            if (fabs(a[i * k + j] - a[i * k + j + m * k]) > 0.0001f)
            {
                printf("%dth row, %dth col; my %f; openblas %f; diff %f\n", i, j, a[i * k + j], a[i * k + j + m * k], fabs(a[i * k + j] - a[i * k + j + m * k]));
            }
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            a[i + j * m] = a[i * k + j + m * k];
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(b[i * n + j] - b[i * n + j + n * k]) > 0.0001f)
            {
                printf("%dth row, %dth col; my %f; openblas %f; diff %f\n", i, j, b[i * n + j], b[i * n + j + n * k], fabs(b[i * n + j] - b[i * n + j + n * k]));
            }
        }
    }

    // execute openblas
    openblas_set_num_threads(1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a + m * k, k, b + n * k, n, 0, c + m * n, n);

    // execute pidan's kernel
    sgemm_kernel_6x16(a, b, c, m, n, k, n);
    printf("----------------results------------------\n");
    // not precise but useful
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

    __aligned_free(a);
    __aligned_free(b);
    __aligned_free(c);

    // test GFLOPS
    //  int i;
    //  // warm up
    //  for (i = 0; i < loop_time; i++)
    //  {
    //      // sgemm_asm_4x8(m, n, k, 0, a, b, 0, c, n);
    //      // sgemm_kernel_4x8(a, b, c, m, n, k, n);
    //      // sgemm_asm_6x16(m, n, k, 0, a, b, 0, c, n);
    //      sgemm_kernel_6x16(a, b, c, m, n, k, n);
    //  }

    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // for (i = 0; i < loop_time; i++)
    // {
    //     // sgemm_asm_4x8(m, n, k, 0, a, b, 0, c, n);
    //     // sgemm_kernel_4x8(a, b, c, m, n, k, n);
    //     // sgemm_asm_6x16(m, n, k, 0, a, b, 0, c, n);
    //     sgemm_kernel_6x16(a, b, c, m, n, k, n);
    // }
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // t = get_time(&start, &end) / loop_time;
    // gflops = (double)comp / t * 1e-9;

    // printf("sgemm_asm_6x16(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n, k, t * 1e6, gflops);
    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         printf("%lf;", c[i * n + j]);
    //     }
    //     printf("\n");
    // }
}