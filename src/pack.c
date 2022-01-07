

void sgemm_pack_a_mc_kc(int mc, int nc, int kc, const float *A, int lda, float *A_pack)
{
    // v0 no optimize
    for (int j = 0; j < kc; j++)
    {
        for (int i = 0; i < mc; i++)
        {
            A_pack[i + j * mc] = A[i * lda + j];
        }
    }
    // v1 intrinsics
    // v2 asm
}
void sgemm_pack_b_kc_nc(int mc, int nc, int kc, const float *B, int ldb, float *B_pack)
{
    // v0 no optimize
    for (int i = 0; i < kc; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            B_pack[i * nc + j] = B[i * ldb + j];
        }
    }
    // v1 intrinsics
    // v2 asm
}