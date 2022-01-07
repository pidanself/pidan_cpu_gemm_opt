.macro PIDAN_C_6x16_LD
    leaq (%rdx, %r12, 8), %r13
    leaq (%r13, %r12, 8), %r14

    vmovaps (%rdx), %ymm0
    vmovaps 32(%rdx), %ymm1
    vmovaps (%rdx, %r12, 4), %ymm2
    vmovaps 32(%rdx, %r12, 4), %ymm3   

    vmovaps (%r13), %ymm4
    vmovaps 32(%r13), %ymm5
    vmovaps (%r13, %r12, 4), %ymm6
    vmovaps 32(%r13, %r12, 4), %ymm7  

    vmovaps (%r14), %ymm8
    vmovaps 32(%r14), %ymm9
    vmovaps (%r14, %r12, 4), %ymm10
    vmovaps 32(%r14, %r12, 4), %ymm11
.endm

.macro PIDAN_C_6x16_ST
    vmovaps %ymm0, (%rdx)
    vmovaps %ymm1, 32(%rdx)
    vmovaps %ymm2, (%rdx, %r12, 4)
    vmovaps %ymm3, 32(%rdx, %r12, 4)

    vmovaps %ymm4, (%r13)
    vmovaps %ymm5, 32(%r13)
    vmovaps %ymm6, (%r13, %r12, 4)
    vmovaps %ymm7, 32(%r13, %r12, 4)

    vmovaps %ymm8, (%r14)
    vmovaps %ymm9, 32(%r14)
    vmovaps %ymm10, (%r14, %r12, 4)
    vmovaps %ymm11, 32(%r14, %r12, 4)
.endm

/* %rsi: b ; %rdi: a */
.macro PIDAN_C_m6k4n16
    /* iter0 */
    vmovaps (%rsi), %ymm13
    vmovaps 32(%rsi), %ymm14
    vbroadcastss (%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm0
    vfmadd231ps %ymm12, %ymm14, %ymm1
    vbroadcastss 4(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm2
    vfmadd231ps %ymm12, %ymm14, %ymm3
    vbroadcastss 8(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm4
    vfmadd231ps %ymm12, %ymm14, %ymm5
    vbroadcastss 12(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm6
    vfmadd231ps %ymm12, %ymm14, %ymm7
    vbroadcastss 16(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm8
    vfmadd231ps %ymm12, %ymm14, %ymm9
    vbroadcastss 20(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm10
    vfmadd231ps %ymm12, %ymm14, %ymm11

    /* iter1 */
    vmovaps 64(%rsi), %ymm13
    vmovaps 96(%rsi), %ymm14
    vbroadcastss 24(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm0
    vfmadd231ps %ymm12, %ymm14, %ymm1
    vbroadcastss 28(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm2
    vfmadd231ps %ymm12, %ymm14, %ymm3
    vbroadcastss 32(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm4
    vfmadd231ps %ymm12, %ymm14, %ymm5
    vbroadcastss 36(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm6
    vfmadd231ps %ymm12, %ymm14, %ymm7
    vbroadcastss 40(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm8
    vfmadd231ps %ymm12, %ymm14, %ymm9
    vbroadcastss 44(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm10
    vfmadd231ps %ymm12, %ymm14, %ymm11

    /* iter2 */
    vmovaps 128(%rsi), %ymm13
    vmovaps 160(%rsi), %ymm14
    vbroadcastss 48(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm0
    vfmadd231ps %ymm12, %ymm14, %ymm1
    vbroadcastss 52(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm2
    vfmadd231ps %ymm12, %ymm14, %ymm3
    vbroadcastss 56(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm4
    vfmadd231ps %ymm12, %ymm14, %ymm5
    vbroadcastss 60(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm6
    vfmadd231ps %ymm12, %ymm14, %ymm7
    vbroadcastss 64(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm8
    vfmadd231ps %ymm12, %ymm14, %ymm9
    vbroadcastss 68(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm10
    vfmadd231ps %ymm12, %ymm14, %ymm11

    /* iter3 */
    vmovaps 192(%rsi), %ymm13
    vmovaps 224(%rsi), %ymm14
    vbroadcastss 72(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm0
    vfmadd231ps %ymm12, %ymm14, %ymm1
    vbroadcastss 76(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm2
    vfmadd231ps %ymm12, %ymm14, %ymm3
    vbroadcastss 80(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm4
    vfmadd231ps %ymm12, %ymm14, %ymm5
    vbroadcastss 84(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm6
    vfmadd231ps %ymm12, %ymm14, %ymm7
    vbroadcastss 88(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm8
    vfmadd231ps %ymm12, %ymm14, %ymm9
    vbroadcastss 92(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm10
    vfmadd231ps %ymm12, %ymm14, %ymm11
.endm


/* MR KC NR */
/* m=6 k=? n=16 */
/* A pannel col major, B pannel row major, C row major*/
/* param rdi:      mat a */
/* param rsi:      mat b */
/* param rdx:      mat c */
/* param ecx:      m     */
/* param r8d:      n     */
/* param r9d:      k     */
/* param stack to r12 :      ldc     */
.globl _sgemm_kernel_6x16
_sgemm_kernel_6x16:
    /* push preserved registers*/
    pushq %r12
    pushq %r13
    pushq %r14

    /* get parameters from caller */
    movl 32(%rsp), %r12d
    movslq %r12d, %r12
    movslq %r9d, %r9
    movq %r9, %r8
    subq $3, %r8

    PIDAN_C_6x16_LD /* load c to SIMD regs */
    xorq %rax, %rax /* rax: index for loop a */
.PIDAN.C.m6k4n16.L1:
    cmpq %rax, %r8
    jle .PIDAN.C.IF.REM 
    PIDAN_C_m6k4n16
    addq $96, %rdi
    addq $256, %rsi
    addq $4, %rax
    jmp .PIDAN.C.m6k4n16.L1

.PIDAN.C.IF.REM:
    cmpq %rax, %r9
    jle .PIDAN.C.NO.REM
    vmovaps (%rsi), %ymm13
    vmovaps 32(%rsi), %ymm14
    vbroadcastss (%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm0
    vfmadd231ps %ymm12, %ymm14, %ymm1
    vbroadcastss 4(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm2
    vfmadd231ps %ymm12, %ymm14, %ymm3
    vbroadcastss 8(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm4
    vfmadd231ps %ymm12, %ymm14, %ymm5
    vbroadcastss 12(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm6
    vfmadd231ps %ymm12, %ymm14, %ymm7
    vbroadcastss 16(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm8
    vfmadd231ps %ymm12, %ymm14, %ymm9
    vbroadcastss 20(%rdi), %ymm12
    vfmadd231ps %ymm12, %ymm13, %ymm10
    vfmadd231ps %ymm12, %ymm14, %ymm11
    addq $24, %rdi
    addq $64, %rsi
    addq $1, %rax
    jmp .PIDAN.C.IF.REM

.PIDAN.C.NO.REM:
    PIDAN_C_6x16_ST /* store SIMD regs back to c */
    /* pop preserved registers */
    popq %r14
    popq %r13
    popq %r12
    retq