#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include "pidanGemm.h"
#include "config.h"
#include "float.h"

static double get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

extern void sgemm_m_n(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    float *C);

void sgemm(int m,
           int n,
           int k,
           const float *A,
           const float *B,
           float *C)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int kk = 0; kk < k; kk++)
            {
                C[i * n + j] += A[i * k + kk] * B[kk * n + j];
            }
        }
    }
}

// print OpenBLAS and pidan_cpu_gemm_opt
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

    long comp = 2L * m * k * n;
    int loop_time = (int)(2e11 / comp);

    struct timespec start, end;
    double t, gflops;
    float *a = (float *)__aligned_malloc(2 * m * k * sizeof(float), 32);
    float *b = (float *)__aligned_malloc(2 * n * k * sizeof(float), 32);
    float *c = (float *)__aligned_malloc(2 * m * n * sizeof(float), 32);

    for (int i = 0; i < m * k; i++)
    {
        a[i] = 1;
    }
    for (int i = 0; i < k * n; i++)
    {
        b[i] = 2;
    }
    for (int i = 0; i < m * n; i++)
    {
        c[i] = 0;
    }

    int i;
    // warm up
    for (i = 0; i < loop_time; i++)
    {
        sgemm_m_n(m, n, k, a, b, c);
    }
    loop_time = 20;
    t = DBL_MAX;
    for (i = 0; i < loop_time; i++)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        sgemm_m_n(m, n, k, a, b, c);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        t = minf(t, get_time(&start, &end));
        // printf("time %f\n", t);
    }
    gflops = (double)comp / t * 1e-9;

    printf("sgemm_m_n(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n, k, t * 1e6, gflops);

    t = DBL_MAX;
    openblas_set_num_threads(1);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (i = 0; i < loop_time; i++)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1, a, m, b, k, 0, c, m);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        t = minf(t, get_time(&start, &end));
        // printf("time %f\n", t);
    }

    gflops = (double)comp / t * 1e-9;

    printf("OpenBLAS(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n, k, t * 1e6, gflops);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (i = 0; i < loop_time; i++)
    {
        // sgemm(m, n, k, a, b, c);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    t = get_time(&start, &end) / loop_time;
    gflops = (double)comp / t * 1e-9;

    printf("gemm(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n, k, t * 1e6, gflops);

    return 0;
}