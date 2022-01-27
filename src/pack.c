#include "config.h"
#include <immintrin.h>

// ref: https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2/25627536#25627536
// ref: https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions
void sgemm_pack_a_mc_kc(int mc, int kc, const float *A, int lda, float *A_pack)
{
    int mr = PIDAN_MR;
    // v0 no optimize
    // for (int i = 0; i < mc; i += mr)
    // {
    //     for (int mm = 0; mm < mr; mm++)
    //     {
    //         for (int kk = 0; kk < kc; kk++)
    //         {
    //             A_pack[i * kc + mm + kk * mr] = A[(i + mm) * lda + kk];
    //         }
    //     }
    // }
    // return;

    /* -----------------I'm line:)---------------------- */
    // v1 intrinsics
    int k_itr = kc / 16;
    int k_rem = kc % 16;
    int m_itr = mc / mr;
    int m_rem = mc % mr;
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
    __m256 ymm0t, ymm1t, ymm2t, ymm3t, ymm4t, ymm5t;
    const float *tempA;
    // m_itr
    for (int j = 0; j < m_itr; j++)
    {
        tempA = A + j * mr * lda;
        // k_itr
        for (int i = 0; i < k_itr; i++)
        {
            // transpose 6x16 and will get col-major 6x16
            _mm_prefetch(tempA + 16, _MM_HINT_NTA); // prefetch type is not important
            _mm_prefetch(tempA + 16 + lda, _MM_HINT_NTA);
            _mm_prefetch(tempA + 16 + 2 * lda, _MM_HINT_NTA);
            _mm_prefetch(tempA + 16 + 3 * lda, _MM_HINT_NTA);
            _mm_prefetch(tempA + 16 + 4 * lda, _MM_HINT_NTA);
            _mm_prefetch(tempA + 16 + 5 * lda, _MM_HINT_NTA);

            // load
            ymm0 = _mm256_loadu_ps(tempA);
            ymm1 = _mm256_loadu_ps(tempA + lda);
            ymm2 = _mm256_loadu_ps(tempA + 2 * lda);
            ymm3 = _mm256_loadu_ps(tempA + 3 * lda);
            ymm4 = _mm256_loadu_ps(tempA + 4 * lda);
            ymm5 = _mm256_loadu_ps(tempA + 5 * lda);
            // reorder
            ymm0t = _mm256_unpacklo_ps(ymm0, ymm1);
            ymm1t = _mm256_unpackhi_ps(ymm0, ymm1);
            ymm2t = _mm256_unpacklo_ps(ymm2, ymm3);
            ymm3t = _mm256_unpackhi_ps(ymm2, ymm3);
            ymm4t = _mm256_unpacklo_ps(ymm4, ymm5);
            ymm5t = _mm256_unpackhi_ps(ymm4, ymm5);

            ymm0 = _mm256_shuffle_ps(ymm0t, ymm2t, _MM_SHUFFLE(1, 0, 1, 0));
            ymm1 = _mm256_shuffle_ps(ymm2t, ymm4t, _MM_SHUFFLE(3, 2, 3, 2));
            ymm2 = _mm256_shuffle_ps(ymm5t, ymm1t, _MM_SHUFFLE(3, 2, 1, 0));
            ymm3 = _mm256_shuffle_ps(ymm4t, ymm0t, _MM_SHUFFLE(3, 2, 1, 0));
            ymm4 = _mm256_shuffle_ps(ymm1t, ymm3t, _MM_SHUFFLE(1, 0, 1, 0));
            ymm5 = _mm256_shuffle_ps(ymm3t, ymm5t, _MM_SHUFFLE(3, 2, 3, 2));

            ymm0t = _mm256_permute2f128_ps(ymm0, ymm3, 0x20);
            ymm1t = _mm256_permute2f128_ps(ymm1, ymm4, 0x20);
            ymm2t = _mm256_permute2f128_ps(ymm2, ymm5, 0x20);
            ymm3t = _mm256_permute2f128_ps(ymm0, ymm3, 0x31);
            ymm4t = _mm256_permute2f128_ps(ymm1, ymm4, 0x31);
            ymm5t = _mm256_permute2f128_ps(ymm2, ymm5, 0x31);

            // store
            _mm256_storeu_ps(A_pack, ymm0t);
            _mm256_storeu_ps(A_pack + 8, ymm1t);
            _mm256_storeu_ps(A_pack + 16, ymm2t);
            _mm256_storeu_ps(A_pack + 24, ymm3t);
            _mm256_storeu_ps(A_pack + 32, ymm4t);
            _mm256_storeu_ps(A_pack + 40, ymm5t);

            A_pack += mr * 8;
            tempA += 8;

            // load
            ymm0 = _mm256_loadu_ps(tempA);
            ymm1 = _mm256_loadu_ps(tempA + lda);
            ymm2 = _mm256_loadu_ps(tempA + 2 * lda);
            ymm3 = _mm256_loadu_ps(tempA + 3 * lda);
            ymm4 = _mm256_loadu_ps(tempA + 4 * lda);
            ymm5 = _mm256_loadu_ps(tempA + 5 * lda);
            // reorder
            ymm0t = _mm256_unpacklo_ps(ymm0, ymm1);
            ymm1t = _mm256_unpackhi_ps(ymm0, ymm1);
            ymm2t = _mm256_unpacklo_ps(ymm2, ymm3);
            ymm3t = _mm256_unpackhi_ps(ymm2, ymm3);
            ymm4t = _mm256_unpacklo_ps(ymm4, ymm5);
            ymm5t = _mm256_unpackhi_ps(ymm4, ymm5);

            ymm0 = _mm256_shuffle_ps(ymm0t, ymm2t, _MM_SHUFFLE(1, 0, 1, 0));
            ymm1 = _mm256_shuffle_ps(ymm2t, ymm4t, _MM_SHUFFLE(3, 2, 3, 2));
            ymm2 = _mm256_shuffle_ps(ymm5t, ymm1t, _MM_SHUFFLE(3, 2, 1, 0));
            ymm3 = _mm256_shuffle_ps(ymm4t, ymm0t, _MM_SHUFFLE(3, 2, 1, 0));
            ymm4 = _mm256_shuffle_ps(ymm1t, ymm3t, _MM_SHUFFLE(1, 0, 1, 0));
            ymm5 = _mm256_shuffle_ps(ymm3t, ymm5t, _MM_SHUFFLE(3, 2, 3, 2));

            ymm0t = _mm256_permute2f128_ps(ymm0, ymm3, 0x20);
            ymm1t = _mm256_permute2f128_ps(ymm1, ymm4, 0x20);
            ymm2t = _mm256_permute2f128_ps(ymm2, ymm5, 0x20);
            ymm3t = _mm256_permute2f128_ps(ymm0, ymm3, 0x31);
            ymm4t = _mm256_permute2f128_ps(ymm1, ymm4, 0x31);
            ymm5t = _mm256_permute2f128_ps(ymm2, ymm5, 0x31);

            // store
            _mm256_storeu_ps(A_pack, ymm0t);
            _mm256_storeu_ps(A_pack + 8, ymm1t);
            _mm256_storeu_ps(A_pack + 16, ymm2t);
            _mm256_storeu_ps(A_pack + 24, ymm3t);
            _mm256_storeu_ps(A_pack + 32, ymm4t);
            _mm256_storeu_ps(A_pack + 40, ymm5t);
            A_pack += mr * 8;
            tempA += 8;
        }

        // k_rem
        for (int i = 0; i < k_rem; i++)
        {
            A_pack[0] = tempA[0];
            A_pack[1] = tempA[lda];
            A_pack[2] = tempA[2 * lda];
            A_pack[3] = tempA[3 * lda];
            A_pack[4] = tempA[4 * lda];
            A_pack[5] = tempA[5 * lda];
            A_pack += 6;
            tempA++;
        }
    }
    //目前只保证正确性，不考虑性能,当需要做下面计算时，性能势必会有损耗；若要更通用，增加kernel或对此处的打包进行优化
    // m_rem
    tempA = A + m_itr * mr * lda;
    for (int i = 0; i < m_rem; i++)
    {
        for (int j = 0; j < kc; j++)
        {
            A_pack[i + j * m_rem] = tempA[i * lda + j];
        }
    }
    return;

    /* -----------------I'm line:)---------------------- */
    // v2 asm
}
void sgemm_pack_b_kc_nc(int nc, int kc, const float *B, int ldb, float *B_pack)
{
    int nr = PIDAN_NR;
    // v0 no optimize
    // for (int i = 0; i < nc; i += nr)
    // {
    //     for (int kk = 0; kk < kc; kk++)
    //     {
    //         for (int nn = 0; nn < nc; nn++)
    //         {
    //             B_pack[i * kc + nn + kk * nr] = B[i + nn + kk * ldb];
    //         }
    //     }
    // }

    /* -----------------I'm line:)---------------------- */
    // v1 intrinsics
    const float *tempB = B;
    int k_itr = kc / 8;
    int k_rem = kc % 8;
    int n_itr = nc / nr;
    int n_rem = nc % nr;
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    // n_itr
    for (int i = 0; i < n_itr; i++)
    {
        tempB = B + i * nr;
        // k_itr
        for (int j = 0; j < k_itr; j++)
        {
            // _mm_prefetch(tempB + 8 * ldb, _MM_HINT_NTA); // prefetch type is not important
            // _mm_prefetch(tempB + 9 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 10 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 11 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 12 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 13 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 14 * ldb, _MM_HINT_NTA);
            // _mm_prefetch(tempB + 15 * ldb, _MM_HINT_NTA);

            ymm0 = _mm256_loadu_ps(tempB);
            ymm1 = _mm256_loadu_ps(tempB + 8);
            ymm2 = _mm256_loadu_ps(tempB + ldb);
            ymm3 = _mm256_loadu_ps(tempB + ldb + 8);
            ymm4 = _mm256_loadu_ps(tempB + 2 * ldb);
            ymm5 = _mm256_loadu_ps(tempB + 2 * ldb + 8);
            ymm6 = _mm256_loadu_ps(tempB + 3 * ldb);
            ymm7 = _mm256_loadu_ps(tempB + 3 * ldb + 8);
            ymm8 = _mm256_loadu_ps(tempB + 4 * ldb);
            ymm9 = _mm256_loadu_ps(tempB + 4 * ldb + 8);
            ymm10 = _mm256_loadu_ps(tempB + 5 * ldb);
            ymm11 = _mm256_loadu_ps(tempB + 5 * ldb + 8);
            ymm12 = _mm256_loadu_ps(tempB + 6 * ldb);
            ymm13 = _mm256_loadu_ps(tempB + 6 * ldb + 8);
            ymm14 = _mm256_loadu_ps(tempB + 7 * ldb);
            ymm15 = _mm256_loadu_ps(tempB + 7 * ldb + 8);
            tempB += 8 * ldb;
            _mm256_storeu_ps(B_pack, ymm0);
            _mm256_storeu_ps(B_pack + 8, ymm1);
            _mm256_storeu_ps(B_pack + 16, ymm2);
            _mm256_storeu_ps(B_pack + 24, ymm3);
            _mm256_storeu_ps(B_pack + 32, ymm4);
            _mm256_storeu_ps(B_pack + 40, ymm5);
            _mm256_storeu_ps(B_pack + 48, ymm6);
            _mm256_storeu_ps(B_pack + 56, ymm7);
            _mm256_storeu_ps(B_pack + 64, ymm8);
            _mm256_storeu_ps(B_pack + 72, ymm9);
            _mm256_storeu_ps(B_pack + 80, ymm10);
            _mm256_storeu_ps(B_pack + 88, ymm11);
            _mm256_storeu_ps(B_pack + 96, ymm12);
            _mm256_storeu_ps(B_pack + 104, ymm13);
            _mm256_storeu_ps(B_pack + 112, ymm14);
            _mm256_storeu_ps(B_pack + 120, ymm15);
            B_pack += 128;
        }
        // k_rem
        for (int j = 0; j < k_rem; j++)
        {
            ymm0 = _mm256_loadu_ps(tempB);
            ymm1 = _mm256_loadu_ps(tempB + 8);
            _mm256_storeu_ps(B_pack, ymm0);
            _mm256_storeu_ps(B_pack + 8, ymm1);
            B_pack += 16;
            tempB += ldb;
        }
    }
    // n_rem
    tempB = B + n_itr * nr;
    for (int i = 0; i < n_rem; i++)
    {
        for (int j = 0; j < kc; j++)
        {
            B_pack[i + j * n_rem] = tempB[i + j * ldb];
        }
    }
    /* -----------------I'm line:)---------------------- */
    // v2 asm
}

// abandon
void sgemm_pack_b_kc_nc_v0(int nc, int kc, const float *B, int ldb, float *B_pack)
{
    int nr = PIDAN_NR;
    // v0 no optimize
    for (int i = 0; i < nc; i += nr)
    {
        for (int kk = 0; kk < kc; kk++)
        {
            for (int nn = 0; nn < nc; nn++)
            {
                B_pack[i * kc + nn + kk * nr] = B[i + nn + kk * ldb];
            }
        }
    }
}