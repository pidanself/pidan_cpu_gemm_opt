#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void sgemm_kernel_4x8(const float *A, const float *B, float *C, int m, int n, int k, int ldc);
extern void sgemm_asm_4x8(int m, int n, int k,
                          float alpha,
                          const float *A, const float *B,
                          float beta,
                          float *C, int ldc);
extern void sgemm_asm_6x16(int m, int n, int k,
                           float alpha,
                           const float *A, const float *B,
                           float beta,
                           float *C, int ldc);
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
    float *a = (float *)aligned_alloc(32, 1024 * sizeof(float));
    float *b = (float *)aligned_alloc(32, 1024 * sizeof(float));
    float *c = (float *)aligned_alloc(32, 1024 * sizeof(float));

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
        // sgemm_asm_4x8(m, n, k, 0, a, b, 0, c, n);
        // sgemm_kernel_4x8(a, b, c, m, n, k, n);
        // sgemm_asm_6x16(m, n, k, 0, a, b, 0, c, n);
        sgemm_kernel_6x16(a, b, c, m, n, k, n);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (i = 0; i < loop_time; i++)
    {
        // sgemm_asm_4x8(m, n, k, 0, a, b, 0, c, n);
        // sgemm_kernel_4x8(a, b, c, m, n, k, n);
        // sgemm_asm_6x16(m, n, k, 0, a, b, 0, c, n);
        sgemm_kernel_6x16(a, b, c, m, n, k, n);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    t = get_time(&start, &end) / loop_time;
    gflops = (double)comp / t * 1e-9;

    printf("sgemm_asm_6x16(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n, k, t * 1e6, gflops);
    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         printf("%lf;", c[i * n + j]);
    //     }
    //     printf("\n");
    // }
}