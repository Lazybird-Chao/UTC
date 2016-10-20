	.file	"shmem_2D_heat.c"
	.text
	.p2align 4,,15
	.globl	gettime
	.type	gettime, @function
gettime:
.LFB53:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	xorl	%esi, %esi
	movq	%rsp, %rdi
	call	gettimeofday
	movq	(%rsp), %rax
	imulq	$1000000, %rax, %rax
	addq	8(%rsp), %rax
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	cvtsi2sdq	%rax, %xmm0
	ret
	.cfi_endproc
.LFE53:
	.size	gettime, .-gettime
	.p2align 4,,15
	.globl	dt
	.type	dt, @function
dt:
.LFB54:
	.cfi_startproc
	movsd	(%rdi), %xmm0
	subsd	(%rsi), %xmm0
	ret
	.cfi_endproc
.LFE54:
	.size	dt, .-dt
	.p2align 4,,15
	.globl	get_start
	.type	get_start, @function
get_start:
.LFB61:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	cvtsi2sd	HEIGHT(%rip), %xmm0
	movl	%edi, %ebx
	call	floor
	cvttsd2si	%xmm0, %eax
	cltd
	idivl	p(%rip)
	cmpl	%ebx, %edx
	jg	.L9
	imull	%ebx, %eax
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	addl	%edx, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L9:
	.cfi_restore_state
	addl	$1, %eax
	imull	%ebx, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE61:
	.size	get_start, .-get_start
	.p2align 4,,15
	.globl	enforce_bc_par
	.type	enforce_bc_par, @function
enforce_bc_par:
.LFB58:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movl	%esi, %r14d
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	movq	%rdi, %r13
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	movl	%ecx, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movl	%edx, %ebx
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	cvtsi2sd	WIDTH(%rip), %xmm1
	movapd	%xmm1, %xmm0
	movsd	%xmm1, 8(%rsp)
	mulsd	.LC0(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movsd	8(%rsp), %xmm1
	subl	$1, %eax
	cmpl	%ebx, %eax
	je	.L18
.L11:
	testl	%ebx, %ebx
	movapd	%xmm1, %xmm0
	jle	.L16
	testl	%ebp, %ebp
	jg	.L13
.L16:
	call	floor
	cvttsd2si	%xmm0, %r12d
.L15:
	movl	%r14d, %edi
	call	get_start
	movl	%ebp, %ecx
	subl	%eax, %ecx
	imull	%r12d, %ecx
	addl	%ebx, %ecx
	movslq	%ecx, %rcx
	movl	$0x00000000, 0(%r13,%rcx,4)
.L10:
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L13:
	.cfi_restore_state
	call	floor
	cvttsd2si	%xmm0, %r12d
	leal	-1(%r12), %eax
	cmpl	%eax, %ebx
	jge	.L15
	cvtsi2sd	HEIGHT(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	subl	$1, %eax
	cmpl	%eax, %ebp
	jge	.L15
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L18:
	testl	%ebp, %ebp
	jne	.L11
	cltq
	movl	$0x44098000, 0(%r13,%rax,4)
	addq	$16, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE58:
	.size	enforce_bc_par, .-enforce_bc_par
	.p2align 4,,15
	.globl	get_end
	.type	get_end, @function
get_end:
.LFB62:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movl	%edi, %ebx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	cvtsi2sd	HEIGHT(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movl	%ebx, %edi
	cltd
	idivl	p(%rip)
	cmpl	%ebx, %edx
	movl	%eax, %ebp
	jg	.L23
	call	get_start
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	leal	-1(%rax,%rbp), %eax
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L23:
	.cfi_restore_state
	call	get_start
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	addl	%ebp, %eax
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE62:
	.size	get_end, .-get_end
	.p2align 4,,15
	.globl	init_domain
	.type	init_domain, @function
init_domain:
.LFB60:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movl	%esi, %r12d
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	movl	%esi, %edi
	call	get_start
	movl	%r12d, %edi
	movl	%eax, %ebp
	call	get_end
	cmpl	%eax, %ebp
	movl	%eax, %r12d
	jg	.L24
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %ecx
	leal	1(%r12), %r9d
	xorl	%r8d, %r8d
	subl	%ebp, %r9d
	.p2align 4,,10
	.p2align 3
.L26:
	movslq	%r8d, %rax
	leaq	(%rbx,%rax,8), %rsi
	xorl	%eax, %eax
	jmp	.L28
	.p2align 4,,10
	.p2align 3
.L27:
	movq	(%rsi), %rdx
	movl	$0x00000000, (%rdx,%rax,4)
	addq	$1, %rax
.L28:
	cmpl	%eax, %ecx
	jg	.L27
	addl	$1, %r8d
	cmpl	%r9d, %r8d
	jne	.L26
.L24:
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE60:
	.size	init_domain, .-init_domain
	.p2align 4,,15
	.globl	get_convergence_sqd
	.type	get_convergence_sqd, @function
get_convergence_sqd:
.LFB56:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movl	%edx, %r13d
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %rbx
	movl	%edx, %edi
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	call	get_start
	movl	%r13d, %edi
	movl	%eax, %r12d
	call	get_end
	cvtsi2sd	WIDTH(%rip), %xmm0
	movl	%eax, %r13d
	call	floor
	cmpl	%r13d, %r12d
	cvttsd2si	%xmm0, %edx
	xorps	%xmm0, %xmm0
	jg	.L31
	leal	1(%r13), %r10d
	xorps	%xmm0, %xmm0
	movslq	%edx, %rsi
	xorl	%r9d, %r9d
	salq	$2, %rsi
	subl	%r12d, %r10d
	.p2align 4,,10
	.p2align 3
.L32:
	movslq	%r9d, %rcx
	xorl	%eax, %eax
	imulq	%rsi, %rcx
	leaq	0(%rbp,%rcx), %r8
	addq	%rbx, %rcx
	jmp	.L34
	.p2align 4,,10
	.p2align 3
.L33:
	movss	(%r8,%rax,4), %xmm1
	subss	(%rcx,%rax,4), %xmm1
	unpcklps	%xmm0, %xmm0
	addq	$1, %rax
	cvtps2pd	%xmm0, %xmm0
	unpcklps	%xmm1, %xmm1
	cvtps2pd	%xmm1, %xmm1
	mulsd	%xmm1, %xmm1
	addsd	%xmm1, %xmm0
	unpcklpd	%xmm0, %xmm0
	cvtpd2ps	%xmm0, %xmm0
.L34:
	cmpl	%eax, %edx
	jg	.L33
	addl	$1, %r9d
	cmpl	%r10d, %r9d
	jne	.L32
.L31:
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE56:
	.size	get_convergence_sqd, .-get_convergence_sqd
	.p2align 4,,15
	.globl	get_val_par
	.type	get_val_par, @function
get_val_par:
.LFB59:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdi, %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rsi, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movl	%ecx, %r12d
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	%r9d, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movslq	%r8d, %rbx
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	cvtsi2sd	WIDTH(%rip), %xmm1
	movapd	%xmm1, %xmm0
	movsd	%xmm1, 8(%rsp)
	movq	%rdx, 16(%rsp)
	mulsd	.LC0(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movsd	8(%rsp), %xmm1
	subl	$1, %eax
	cmpl	%ebx, %eax
	je	.L51
.L38:
	testl	%ebx, %ebx
	jle	.L43
	testl	%ebp, %ebp
	jle	.L43
	movapd	%xmm1, %xmm0
	call	floor
	cvttsd2si	%xmm0, %r13d
	xorps	%xmm2, %xmm2
	leal	-1(%r13), %eax
	cmpl	%eax, %ebx
	jge	.L39
	movss	%xmm2, 8(%rsp)
	cvtsi2sd	HEIGHT(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movss	8(%rsp), %xmm2
	subl	$1, %eax
	cmpl	%eax, %ebp
	jge	.L39
	movl	%r12d, %edi
	call	get_start
	cmpl	%eax, %ebp
	movss	8(%rsp), %xmm2
	jge	.L40
	testl	%r12d, %r12d
	je	.L39
	movss	(%r15,%rbx,4), %xmm2
	.p2align 4,,10
	.p2align 3
.L39:
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	movaps	%xmm2, %xmm0
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L43:
	.cfi_restore_state
	xorps	%xmm2, %xmm2
	jmp	.L39
	.p2align 4,,10
	.p2align 3
.L51:
	testl	%ebp, %ebp
	movss	.LC1(%rip), %xmm2
	jne	.L38
	jmp	.L39
	.p2align 4,,10
	.p2align 3
.L40:
	movl	%r12d, %edi
	movss	%xmm2, 8(%rsp)
	movl	%eax, 28(%rsp)
	call	get_end
	cmpl	%eax, %ebp
	movss	8(%rsp), %xmm2
	movl	28(%rsp), %edx
	jle	.L41
	movl	p(%rip), %eax
	subl	$1, %eax
	cmpl	%r12d, %eax
	je	.L39
	movq	16(%rsp), %rax
	movss	(%rax,%rbx,4), %xmm2
	jmp	.L39
	.p2align 4,,10
	.p2align 3
.L41:
	subl	%edx, %ebp
	imull	%ebp, %r13d
	addl	%r13d, %ebx
	movslq	%ebx, %rbx
	movss	(%r14,%rbx,4), %xmm2
	jmp	.L39
	.cfi_endproc
.LFE59:
	.size	get_val_par, .-get_val_par
	.p2align 4,,15
	.globl	jacobi
	.type	jacobi, @function
jacobi:
.LFB57:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rcx, %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%rdx, %r13
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	%r8d, %ebx
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	movl	p(%rip), %eax
	movq	%rsi, 8(%rsp)
	movl	%r9d, 40(%rsp)
	cmpl	$1, %eax
	jle	.L53
	movl	my_rank(%rip), %ebp
	subl	$1, %eax
	cmpl	%ebp, %eax
	jg	.L61
.L54:
	call	shmem_barrier_all
	movl	my_rank(%rip), %ebp
	testl	%ebp, %ebp
	jle	.L55
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %edx
	leal	-1(%rbp), %ecx
	movq	%r12, %rsi
	movq	%r15, %rdi
	movslq	%edx, %rdx
	call	shmem_float_put
.L55:
	call	shmem_barrier_all
.L53:
	cmpl	40(%rsp), %ebx
	jg	.L52
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movl	%ebx, %esi
	notl	%esi
	movl	%ebx, %r14d
	movl	%esi, 44(%rsp)
	.p2align 4,,10
	.p2align 3
.L57:
	movl	44(%rsp), %ecx
	leal	1(%r14), %esi
	xorl	%ebx, %ebx
	movl	%esi, 28(%rsp)
	addl	%esi, %ecx
	movl	%ecx, 32(%rsp)
	leal	-1(%r14), %ecx
	movl	%ecx, 36(%rsp)
	jmp	.L59
	.p2align 4,,10
	.p2align 3
.L58:
	imull	32(%rsp), %eax
	movq	8(%rsp), %rdi
	movl	my_rank(%rip), %ebp
	leal	-1(%rbx), %r8d
	movl	%r14d, %r9d
	movq	%r15, %rdx
	movq	%r12, %rsi
	movl	%ebp, %ecx
	addl	%ebx, %eax
	cltq
	leaq	(%rdi,%rax,4), %r10
	movq	%r13, %rdi
	movq	%r10, 16(%rsp)
	call	get_val_par
	leal	1(%rbx), %eax
	movl	%r14d, %r9d
	movl	%ebp, %ecx
	movq	%r15, %rdx
	movq	%r12, %rsi
	movq	%r13, %rdi
	movl	%eax, %r8d
	movl	%eax, 24(%rsp)
	movss	%xmm0, 4(%rsp)
	call	get_val_par
	movss	4(%rsp), %xmm1
	movl	%ebx, %r8d
	addss	%xmm0, %xmm1
	movl	36(%rsp), %r9d
	movl	%ebp, %ecx
	movq	%r15, %rdx
	movq	%r12, %rsi
	movq	%r13, %rdi
	movss	%xmm1, 4(%rsp)
	call	get_val_par
	addss	4(%rsp), %xmm0
	movl	%ebx, %r8d
	movl	%ebp, %ecx
	movl	28(%rsp), %r9d
	movq	%r15, %rdx
	movq	%r12, %rsi
	movq	%r13, %rdi
	movss	%xmm0, 4(%rsp)
	call	get_val_par
	addss	4(%rsp), %xmm0
	movl	%ebx, %edx
	movl	%r14d, %ecx
	movq	16(%rsp), %r10
	movq	8(%rsp), %rdi
	movl	%ebp, %esi
	unpcklps	%xmm0, %xmm0
	cvtps2pd	%xmm0, %xmm0
	mulsd	.LC3(%rip), %xmm0
	unpcklpd	%xmm0, %xmm0
	cvtpd2ps	%xmm0, %xmm2
	movss	%xmm2, (%r10)
	call	enforce_bc_par
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	movl	24(%rsp), %eax
	movl	%eax, %ebx
	cvttsd2si	%xmm0, %eax
.L59:
	cmpl	%eax, %ebx
	jl	.L58
	movl	28(%rsp), %edx
	cmpl	%edx, 40(%rsp)
	jl	.L52
	movl	%edx, %r14d
	jmp	.L57
.L52:
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L61:
	.cfi_restore_state
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %eax
	movl	112(%rsp), %esi
	leal	1(%rbp), %ecx
	movq	%r13, %rdi
	subl	$1, %esi
	movslq	%eax, %rdx
	imull	%esi, %eax
	cltq
	leaq	(%r12,%rax,4), %rsi
	call	shmem_float_put
	jmp	.L54
	.cfi_endproc
.LFE57:
	.size	jacobi, .-jacobi
	.p2align 4,,15
	.globl	get_num_rows
	.type	get_num_rows, @function
get_num_rows:
.LFB63:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movl	%edi, %ebx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	call	get_end
	movl	%ebx, %edi
	movl	%eax, %ebp
	call	get_start
	leal	1(%rbp), %edx
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	subl	%eax, %edx
	movl	%edx, %eax
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE63:
	.size	get_num_rows, .-get_num_rows
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC5:
	.string	"Option -%c requires an operand\n"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC6:
	.string	"Unrecognized option: -%c\n"
.LC7:
	.string	"e:h:m:tw:v"
	.section	.rodata.str1.8
	.align 8
.LC8:
	.string	"Usage: oshrun -np <np> %s -h <nrows> -w <ncolumns> -m <method>\n"
	.align 8
.LC9:
	.string	"Using defaults: -h 20 -w 20 -m 2"
	.align 8
.LC10:
	.string	"proc %d contains (%d) rows %d to %d\n"
	.section	.rodata.str1.1
.LC11:
	.string	"iteration: %d\n"
.LC12:
	.string	"L2 = %f\n"
	.section	.rodata.str1.8
	.align 8
.LC14:
	.string	"Estimated time to convergence in %d iterations using %d processors on a %dx%d grid is %f seconds\n"
	.section	.rodata.str1.1
.LC15:
	.string	"%f\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB55:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	%edi, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	call	shmem_init
	call	shmem_n_pes
	movl	%eax, p(%rip)
	call	shmem_my_pe
	movl	$0, 84(%rsp)
	movl	%eax, my_rank(%rip)
.L65:
	movl	$.LC7, %edx
	movq	%rbx, %rsi
	movl	%ebp, %edi
	call	getopt
	cmpl	$-1, %eax
	je	.L117
	subl	$58, %eax
	cmpl	$61, %eax
	ja	.L65
	.p2align 4,,2
	jmp	*.L68(,%rax,8)
	.section	.rodata
	.align 8
	.align 4
.L68:
	.quad	.L67
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L69
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L70
	.quad	.L65
	.quad	.L65
	.quad	.L71
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L72
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L65
	.quad	.L73
	.quad	.L74
	.section	.text.startup
.L73:
	addl	$1, 84(%rsp)
	jmp	.L65
.L74:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol
	movl	%eax, WIDTH(%rip)
	jmp	.L65
.L72:
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	call	strtol
	cmpl	$2, %eax
	je	.L75
	cmpl	$3, %eax
	je	.L76
	subl	$1, %eax
	.p2align 4,,2
	jne	.L65
	movl	$1, meth(%rip)
	jmp	.L65
.L71:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol
	movl	%eax, HEIGHT(%rip)
	jmp	.L65
.L70:
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	call	strtod
	unpcklpd	%xmm0, %xmm0
	cvtpd2ps	%xmm0, %xmm4
	movss	%xmm4, EPSILON(%rip)
	jmp	.L65
.L69:
	movl	my_rank(%rip), %r11d
	testl	%r11d, %r11d
	jne	.L65
	movl	optopt(%rip), %ecx
	movq	stderr(%rip), %rdi
	movl	$.LC6, %edx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk
	jmp	.L65
.L67:
	movl	my_rank(%rip), %r12d
	testl	%r12d, %r12d
	jne	.L65
	movl	optopt(%rip), %ecx
	movq	stderr(%rip), %rdi
	movl	$.LC5, %edx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk
	jmp	.L65
.L117:
	movl	my_rank(%rip), %r10d
	testl	%r10d, %r10d
	jne	.L79
	subl	$1, %ebp
	jle	.L118
.L79:
	movq	$-1, pSync(%rip)
	movq	$-1, pSync+8(%rip)
	movq	$-1, pSync+16(%rip)
	call	shmem_barrier_all
	movl	p(%rip), %eax
	movl	$meth, %esi
	xorl	%r9d, %r9d
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	movq	$pSync, 8(%rsp)
	movl	$1, %edx
	movq	%rsi, %rdi
	movl	%eax, (%rsp)
	call	shmem_broadcast32
	cmpl	$1, meth(%rip)
	jne	.L80
	movq	$jacobi, method(%rip)
.L80:
	movl	my_rank(%rip), %ebx
	movl	%ebx, %edi
	call	get_start
	movl	%ebx, %edi
	movl	%eax, %r15d
	movl	%eax, 40(%rsp)
	call	get_end
	movl	%ebx, %edi
	movl	%eax, %r14d
	movl	%eax, 48(%rsp)
	call	get_num_rows
	movl	84(%rsp), %r9d
	movl	%eax, %ecx
	movl	%eax, 80(%rsp)
	testl	%r9d, %r9d
	je	.L82
	movl	%r14d, %r9d
	movl	%r15d, %r8d
	movl	%ebx, %edx
	movl	$.LC10, %esi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk
.L82:
	movq	stdout(%rip), %rdi
	call	fflush
	movl	80(%rsp), %r15d
	movslq	%r15d, %rbx
	leaq	0(,%rbx,8), %r12
	salq	$2, %rbx
	movq	%r12, %rdi
	call	malloc
	cvtsi2sd	WIDTH(%rip), %xmm0
	movq	%rax, %r14
	movq	%rax, 24(%rsp)
	call	floor
	cvttsd2si	%xmm0, %ebp
	movslq	%ebp, %rbp
	imulq	%rbp, %rbx
	movq	%rbx, %rdi
	call	malloc
	cmpl	$1, %r15d
	movq	%rax, (%r14)
	jle	.L83
	leal	-2(%r15), %edx
	salq	$2, %rbp
	movq	%r14, %rax
	leaq	8(,%rdx,8), %r13
	leaq	(%r14,%r13), %rsi
.L85:
	movq	%rbp, %rdx
	addq	(%rax), %rdx
	addq	$8, %rax
	movq	%rdx, (%rax)
	cmpq	%rsi, %rax
	jne	.L85
	movq	%r12, %rdi
	call	malloc
	movq	%rbx, %rdi
	movq	%rax, %r14
	call	malloc
	leaq	(%r14,%r13), %rcx
	movq	%rax, (%r14)
	movq	%r14, %rax
.L104:
	movq	%rbp, %rdx
	addq	(%rax), %rdx
	addq	$8, %rax
	movq	%rdx, (%rax)
	cmpq	%rcx, %rax
	jne	.L104
.L105:
	movq	%rbp, %rdi
	call	shmem_malloc
	cvtsi2sd	WIDTH(%rip), %xmm0
	movq	%rax, 64(%rsp)
	call	floor
	cvttsd2si	%xmm0, %edi
	movslq	%edi, %rdi
	salq	$2, %rdi
	call	shmem_malloc
	cvtsi2sd	WIDTH(%rip), %xmm0
	movq	%rax, 72(%rsp)
	call	floor
	cvttsd2si	%xmm0, %edi
	movslq	%edi, %rdi
	salq	$2, %rdi
	call	shmem_malloc
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	movq	24(%rsp), %rax
	movl	my_rank(%rip), %esi
	cvttsd2si	%xmm0, %ebx
	movq	(%r14), %r12
	movq	%rax, %rdi
	movq	(%rax), %r15
	call	init_domain
	movl	my_rank(%rip), %esi
	movq	%r14, %rdi
	call	init_domain
	movl	my_rank(%rip), %r8d
	testl	%r8d, %r8d
	jne	.L87
	xorl	%eax, %eax
	call	gettime
	movsd	%xmm0, 88(%rsp)
.L87:
	movl	48(%rsp), %eax
	movslq	%ebx, %rbx
	movl	$1, %r13d
	xorpd	%xmm5, %xmm5
	leaq	0(,%rbx,4), %rbp
	leal	1(%rax), %ebx
	subl	40(%rsp), %ebx
	movsd	%xmm5, 32(%rsp)
.L88:
	xorl	%eax, %eax
	call	gettime
	movl	80(%rsp), %eax
	movl	48(%rsp), %r9d
	movsd	%xmm0, 56(%rsp)
	movl	40(%rsp), %r8d
	movq	72(%rsp), %rcx
	movq	64(%rsp), %rdx
	movl	%eax, (%rsp)
	movq	24(%rsp), %rax
	movq	(%r14), %rsi
	movq	(%rax), %rdi
	call	jacobi
	xorl	%eax, %eax
	call	gettime
	subsd	56(%rsp), %xmm0
	movq	24(%rsp), %rax
	movq	(%r14), %rsi
	movl	my_rank(%rip), %edx
	movq	(%rax), %rdi
	addsd	32(%rsp), %xmm0
	movsd	%xmm0, 32(%rsp)
	call	get_convergence_sqd
	movl	p(%rip), %r9d
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	movl	$local_convergence_sqd, %esi
	movq	$pSync, 8(%rsp)
	movq	$pWrk, (%rsp)
	movl	$1, %edx
	movl	$convergence_sqd, %edi
	movss	%xmm0, local_convergence_sqd(%rip)
	call	shmem_float_sum_to_all
	movl	my_rank(%rip), %esi
	testl	%esi, %esi
	je	.L119
.L91:
	movl	p(%rip), %eax
	movl	$convergence, %esi
	xorl	%r9d, %r9d
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	movq	$pSync, 8(%rsp)
	movl	$1, %edx
	movq	%rsi, %rdi
	movl	%eax, (%rsp)
	call	shmem_broadcast32
	movss	EPSILON(%rip), %xmm0
	ucomiss	convergence(%rip), %xmm0
	jnb	.L95
	movl	48(%rsp), %eax
	cmpl	%eax, 40(%rsp)
	jg	.L96
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	cvttsd2si	%xmm0, %edx
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L97:
	movslq	%edi, %rcx
	xorl	%eax, %eax
	imulq	%rbp, %rcx
	leaq	(%r15,%rcx), %rsi
	addq	%r12, %rcx
	jmp	.L99
	.p2align 4,,10
	.p2align 3
.L98:
	movss	(%rcx,%rax,4), %xmm0
	movss	%xmm0, (%rsi,%rax,4)
	addq	$1, %rax
.L99:
	cmpl	%eax, %edx
	jg	.L98
	addl	$1, %edi
	cmpl	%ebx, %edi
	jne	.L97
.L96:
	call	shmem_barrier_all
	addl	$1, %r13d
	movl	$1374389535, %eax
	movl	$100, %ecx
	imull	%r13d
	movl	%r13d, %eax
	sarl	$31, %eax
	sarl	$5, %edx
	subl	%eax, %edx
	imull	%ecx, %edx
	cmpl	%edx, %r13d
	jne	.L88
	movl	my_rank(%rip), %edi
	testl	%edi, %edi
	jne	.L88
	movl	%r13d, %edx
	movl	$.LC11, %esi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk
	jmp	.L88
.L119:
	sqrtss	convergence_sqd(%rip), %xmm0
	ucomiss	%xmm0, %xmm0
	jp	.L120
.L92:
	cmpl	$1, 84(%rsp)
	movss	%xmm0, convergence(%rip)
	jne	.L91
	cvtss2sd	%xmm0, %xmm0
	movl	$.LC12, %esi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk
	jmp	.L91
.L95:
	movl	my_rank(%rip), %eax
	testl	%eax, %eax
	jne	.L101
	xorl	%eax, %eax
	call	gettime
	movsd	%xmm0, 48(%rsp)
	cvtsi2sd	HEIGHT(%rip), %xmm0
	call	floor
	movsd	%xmm0, 40(%rsp)
	cvtsi2sd	WIDTH(%rip), %xmm0
	call	floor
	movsd	48(%rsp), %xmm1
	movl	$.LC14, %esi
	movapd	%xmm0, %xmm2
	movl	p(%rip), %ecx
	subsd	88(%rsp), %xmm1
	movsd	40(%rsp), %xmm3
	cvttsd2si	%xmm2, %r8d
	movl	$1, %edi
	cvttsd2si	%xmm3, %r9d
	movl	%r13d, %edx
	movl	$1, %eax
	movapd	%xmm1, %xmm0
	divsd	.LC13(%rip), %xmm0
	call	__printf_chk
	movsd	32(%rsp), %xmm0
	movl	$.LC15, %esi
	movl	$1, %edi
	movl	$1, %eax
	divsd	.LC13(%rip), %xmm0
	call	__printf_chk
.L101:
	movq	24(%rsp), %rax
	movq	(%rax), %rdi
	testq	%rdi, %rdi
	je	.L102
	call	free
.L102:
	movq	24(%rsp), %rdi
	call	free
	movq	(%r14), %rdi
	testq	%rdi, %rdi
	je	.L103
	call	free
.L103:
	movq	%r14, %rdi
	call	free
	xorl	%edi, %edi
	call	exit
.L118:
	movq	(%rbx), %rdx
	movl	$.LC8, %esi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk
	movl	$.LC9, %edi
	call	puts
	jmp	.L79
.L76:
	movl	$3, meth(%rip)
	jmp	.L65
.L75:
	movl	$2, meth(%rip)
	jmp	.L65
.L83:
	movq	%r12, %rdi
	salq	$2, %rbp
	call	malloc
	movq	%rbx, %rdi
	movq	%rax, %r14
	call	malloc
	movq	%rax, (%r14)
	jmp	.L105
.L120:
	movss	convergence_sqd(%rip), %xmm0
	call	sqrtf
	jmp	.L92
	.cfi_endproc
.LFE55:
	.size	main, .-main
	.text
	.p2align 4,,15
	.globl	global_to_local
	.type	global_to_local, @function
global_to_local:
.LFB64:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movl	%esi, %ebx
	call	get_start
	subl	%eax, %ebx
	movl	%ebx, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE64:
	.size	global_to_local, .-global_to_local
	.p2align 4,,15
	.globl	f
	.type	f, @function
f:
.LFB65:
	.cfi_startproc
	xorps	%xmm0, %xmm0
	ret
	.cfi_endproc
.LFE65:
	.size	f, .-f
	.comm	my_rank,4,4
	.comm	p,4,4
	.comm	method,8,8
	.comm	local_convergence_sqd,4,4
	.comm	convergence_sqd,4,4
	.comm	convergence,4,4
	.comm	pWrk,4,4
	.comm	pSync,24,16
	.globl	EPSILON
	.data
	.align 4
	.type	EPSILON, @object
	.size	EPSILON, 4
EPSILON:
	.long	1036831949
	.globl	meth
	.align 4
	.type	meth, @object
	.size	meth, 4
meth:
	.long	2
	.globl	HEIGHT
	.align 4
	.type	HEIGHT, @object
	.size	HEIGHT, 4
HEIGHT:
	.long	20
	.globl	WIDTH
	.align 4
	.type	WIDTH, @object
	.size	WIDTH, 4
WIDTH:
	.long	20
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1071644672
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1141473280
	.section	.rodata.cst8
	.align 8
.LC3:
	.long	0
	.long	1070596096
	.align 8
.LC13:
	.long	0
	.long	1093567616
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
