.macro PIDAN_C_6x16_LD
    vmovups (%rdx), %ymm0
    vmovups 32(%rdx), %ymm1
    vmovups (%rdx, %r12, 4), %ymm2
    vmovups 32(%rdx, %r12, 4), %ymm3   

    vmovups (%r13), %ymm4
    vmovups 32(%r13), %ymm5
    vmovups (%r13, %r12, 4), %ymm6
    vmovups 32(%r13, %r12, 4), %ymm7  

    vmovups (%r14), %ymm8
    vmovups 32(%r14), %ymm9
    vmovups (%r14, %r12, 4), %ymm10
    vmovups 32(%r14, %r12, 4), %ymm11
.endm

.macro PIDAN_CLEAN_REG
    vxorps	%ymm0, %ymm0, %ymm0
    vxorps	%ymm1, %ymm1, %ymm1
    vxorps	%ymm2, %ymm2, %ymm2
    vxorps	%ymm3, %ymm3, %ymm3
    vxorps	%ymm4, %ymm4, %ymm4
    vxorps	%ymm5, %ymm5, %ymm5
    vxorps	%ymm6, %ymm6, %ymm6
    vxorps	%ymm7, %ymm7, %ymm7
    vxorps	%ymm8, %ymm8, %ymm8
    vxorps	%ymm9, %ymm9, %ymm9
    vxorps	%ymm10, %ymm10, %ymm10
    vxorps	%ymm11, %ymm11, %ymm11
    vxorps	%ymm12, %ymm12, %ymm12
    vxorps	%ymm13, %ymm13, %ymm13
    vxorps	%ymm14, %ymm14, %ymm14
    vxorps	%ymm15, %ymm15, %ymm15
.endm

.macro PIDAN_C_6x16_ST
    vmovups %ymm0, (%rdx)
    vmovups %ymm1, 32(%rdx)
    vmovups %ymm2, (%rdx, %r12, 4)
    vmovups %ymm3, 32(%rdx, %r12, 4)

    vmovups %ymm4, (%r13)
    vmovups %ymm5, 32(%r13)
    vmovups %ymm6, (%r13, %r12, 4)
    vmovups %ymm7, 32(%r13, %r12, 4)

    vmovups %ymm8, (%r14)
    vmovups %ymm9, 32(%r14)
    vmovups %ymm10, (%r14, %r12, 4)
    vmovups %ymm11, 32(%r14, %r12, 4)
.endm

/* %rsi: b ; %rdi: a */
.macro PIDAN_C_m6k4n16
    /* iter0 */
    vmovups (%rsi), %ymm13
    vmovups 32(%rsi), %ymm14
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
    vmovups 64(%rsi), %ymm13
    vmovups 96(%rsi), %ymm14
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
    vmovups 128(%rsi), %ymm13
    vmovups 160(%rsi), %ymm14
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
    vmovups 192(%rsi), %ymm13
    vmovups 224(%rsi), %ymm14
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

    leaq (%rdx, %r12, 8), %r13
    leaq (%r13, %r12, 8), %r14
    /* PIDAN_C_6x16_LD load c to SIMD regs */
    PIDAN_C_6x16_LD
    prefetcht0 (%rsi)
    prefetcht0 (%rdi)
    xorq %rax, %rax /* rax: index for loop a */
.PIDAN.C.m6k4n16.L1:
    cmpq %rax, %r8
    jle .PIDAN.C.IF.REM 
    prefetcht0 4*64(%rsi)
    prefetcht0 96(%rdi)
    PIDAN_C_m6k4n16
    addq $96, %rdi
    addq $256, %rsi
    addq $4, %rax
    jmp .PIDAN.C.m6k4n16.L1

.PIDAN.C.IF.REM:
    cmpq %rax, %r9
    jle .PIDAN.C.NO.REM
    vmovups (%rsi), %ymm13
    vmovups 32(%rsi), %ymm14
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