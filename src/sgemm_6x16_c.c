
// thre function is just compare to prove vector computing is useful
void sgemm_kernel_6x16_c(const float *A,
                         const float *B,
                         float *C,
                         int m,
                         int n,
                         int k,
                         int ldc)
{
    for (int kk = 0; kk < k; kk++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {

                C[i * ldc + j] += A[i + m * kk] * B[kk * n + j];
            }
        }
    }
}