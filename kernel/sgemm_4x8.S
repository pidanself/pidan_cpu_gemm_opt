.macro PIDAN_C_4x8_LD
    leaq (%rdx, %r12, 8), %r13
    vmovaps (%rdx), %ymm0
    vmovaps (%rdx, %r12, 4), %ymm1
    vmovaps (%r13), %ymm2
    vmovaps (%r13, %r12, 4), %ymm3
.endm

.macro PIDAN_C_4x8_ST
    vmovaps %ymm0, (%rdx)
    vmovaps %ymm1, (%rdx, %r12, 4)
    vmovaps %ymm2, (%r13)
    vmovaps %ymm3, (%r13, %r12, 4)
.endm

.macro PIDAN_C_m4k2n8
    vmovaps (%rsi), %ymm5
    vbroadcastss (%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm0
    vbroadcastss 4(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm1
    vbroadcastss 8(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm2
    vbroadcastss 12(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm3

    vmovaps 32(%rsi), %ymm5
    vbroadcastss 16(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm0
    vbroadcastss 20(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm1
    vbroadcastss 24(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm2
    vbroadcastss 28(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm3
.endm



/* 此kernel的问题在于，一个循环中的fmad计算不超过8条，导致循环间产生RAW依赖，无法将流水线打满 */
/* 同时，访存计算比勉强为1，可能会卡访存 */
/* 目前来看，主要卡在了计算这里，流水线流不起来，没有发挥双端口的优势 */

/* m=4 k=? n=8*/
/* A pannel col major, B pannel row major, C row major*/
/* param rdi:      mat a */
/* param rsi:      mat b */
/* param rdx:      mat c */
/* param ecx:      m     */
/* param r8d:      n     */
/* param r9d:      k     */
/* param stack to r12 :      ldc     */
.globl _sgemm_kernel_4x8
_sgemm_kernel_4x8:
    /* push preserved registers*/
    pushq %r12
    pushq %r13

    /* get parameters from caller */
    /*movq 8(%rsp), %r12*/
    /*movslq %r12d, %r12*/
    movq $8, %r12
    movslq %r9d, %r9
    movq %r9, %r8
    subq $2, %r8

    PIDAN_C_4x8_LD /* load c to SIMD regs */
    xorq %rax, %rax /* rax: index for loop a */
.PIDAN.C.m4k2n8.L1:
    PIDAN_C_m4k2n8
    addq $32, %rdi
    addq $64, %rsi
    addq $2, %rax
    cmpq %rax, %r8 /* index < k */
    jge .PIDAN.C.m4k2n8.L1
    cmpq %r9, %rax
    je .PIDAN.C.NO.REM
.PIDAN.C.REM:
    subq $16, %rdi
    subq $32, %rsi
    vmovaps (%rsi), %ymm5
    vbroadcastss (%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm0
    vbroadcastss 4(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm1
    vbroadcastss 8(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm2
    vbroadcastss 12(%rdi), %ymm4
    vfmadd231ps %ymm5, %ymm4, %ymm3

.PIDAN.C.NO.REM:
    PIDAN_C_4x8_ST /* store SIMD regs back to c */

    /* pop preserved registers */
    popq %r12
    popq %r13
    retq