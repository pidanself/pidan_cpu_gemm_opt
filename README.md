# pidan_cpu_gemm_opt
Try to exceed openblas gemm by using single cpu core

TODO:  
- [x] Macro kernel: 90% cpu perf when datas are in L1 cache
- [x] Pack A matrixs
- [x] Pack B matrixs
- [x] Loop NR : NC
- [x] Loop MR : MC
- [x] Loop NC : N
- [x] Loop KC : K
- [x] Loop MC : M
- [x] Elegant makefile 
- [ ] Test tools       
  - [x]  validity
  - [ ]  performance
- [ ] make 4128 size get 96% cpu perf by tuning 
- [ ] why errors are so big with openblas? Is that normal?
- [ ] how useful prefetch is in sgemm_6x16? 
- [ ] how useful prefetch is in A_pack and B_pack?

