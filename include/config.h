#include <time.h>
#include <stdio.h>
void sgemm_kernel_6x16(const float *A,
                       const float *B,
                       float *C,
                       int m,
                       int n,
                       int k,
                       int ldc);
void sgemm_kernel_6x16_c(const float *A,
                         const float *B,
                         float *C,
                         int m,
                         int n,
                         int k,
                         int ldc);
void sgemm_mc_nc(
    int mc,
    int nc,
    int kc,
    const float *packA,
    const float *packB,
    float *C,
    int ldc);
#define PIDAN_MR 6
#define PIDAN_NR 16

// TODO: 不理解数据按照page size对齐有什么好处
#define PIDAN_PAGE_SIZE 4096

// utils
void getOptimizeMCNCKC(int m, int n, int k, int *mc, int *nc, int *kc);
void *__aligned_malloc(unsigned long required_bytes, unsigned long alignment);
void __aligned_free(void *p);
int min(int a, int b);
double minf(double a, double b);

// pack
void sgemm_pack_a_mc_kc(int mc, int kc, const float *A, int lda, float *A_pack);
void sgemm_pack_b_kc_nc(int nc, int kc, const float *B, int ldb, float *B_pack);
void sgemm_pack_b_kc_nc_v0(int nc, int kc, const float *B, int ldb, float *B_pack);