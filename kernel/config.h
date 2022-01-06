void sgemm_kernel_6x16(const float *A, const float *B, float *C, int m, int n, int k, int ldc);
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