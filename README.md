# pidan_cpu_gemm_opt
Try to exceed openblas gemm by using single cpu core

TODO:  
- [x] Macro kernel: 90% cpu perf when datas are in L1 cache
- [ ] Pack matrixs
- [x] Loop NR : NC
- [x] Loop MR : MC
- [x] Loop NC : N
- [x] Loop KC : K
- [x] Loop MC : M
- [ ] Elegant makefile 
- [ ] Test tools
