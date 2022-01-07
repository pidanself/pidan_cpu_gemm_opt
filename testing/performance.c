#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    long comp = 2L * m * k * 32L;
    int loop_time = (int)(2e11 / comp);

    struct timespec start, end;
    double t, gflops;
    float *a = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc(n * k * sizeof(float));
    float *c = (float *)malloc(m * n * sizeof(float));

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
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (i = 0; i < loop_time; i++)
    {
        sgemm_m_n(m, n, k, a, b, c);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    t = get_time(&start, &end) / loop_time;
    gflops = (double)comp / t * 1e-9;

    printf("sgemm_m_n(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, 24, k, t * 1e6, gflops);

    return 0;
}