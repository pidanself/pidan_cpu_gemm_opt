# pidan_cpu_gemm_opt
Try to exceed openblas gemm by using single cpu core

TODO:  
- [x] Micro kernel: 90% cpu perf when datas are in L1 cache
- [x] Pack A matrixs
- [x] Pack B matrixs
- [x] Loop NR : NC
- [x] Loop MR : MC
- [x] Loop NC : N
- [x] Loop KC : K
- [x] Loop MC : M
- [x] Elegant makefile 
- [x] Test tools       
  - [x]  validity
  - [x]  performance
- [x] make 1152 size get about 90% cpu perf by tuning 
- [ ] why errors are so big with openblas? Is that normal?
- [x] how useful prefetch is in sgemm_6x16? prefetch in sgemm_6x16 improves about 10 Gflops
  - [x] add prefetch
  - [x] why prefetch is useful? reduce necessary cache misses
- [x] how useful prefetch is in A_pack and B_pack? prefetch in B_pack is useless. prefetch in A_pack improves about 3 Gflops
  - [x] add prefetch
  - [x] why prefetch is useful? reduce necessary cache misses

