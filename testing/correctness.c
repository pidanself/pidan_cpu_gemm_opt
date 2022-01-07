#include <stdio.h>
#include <stdlib.h>

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
    sgemm_m_n(m, n, k, a, b, c);

    return 0;
}