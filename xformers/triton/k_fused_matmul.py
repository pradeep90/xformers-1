# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl

# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications


# fmt: off
@triton.autotune(
    configs=[
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 32 , 'BLOCK_COL': 64}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 256}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 256}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 32}, num_stages=4, num_warps=4),
        ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    OUT, INPUT, WEIGHT, BIAS, ACT_INPUTS,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_dm, stride_am,
    stride_wn, stride_wk,
    stride_aim,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """

    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_ROW"], META["GROUP_ROW"]
    BLOCK_N, BLOCK_K = META["BLOCK_COL"], META["BLOCK_K"]

    # programs are grouped together to improve L2 hit rate
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)

    # col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    Input_ptrs = INPUT + rm[:, None] * stride_am + rk[None, :]
    Weight_ptrs = WEIGHT + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if META["BIAS"]:
        bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # block level matrix multiplication
    for _ in range(K, 0, -BLOCK_K):
        a = tl.load(Input_ptrs)
        w = tl.load(Weight_ptrs)
        acc += tl.dot(a, w).to(tl.float32)

        Input_ptrs += BLOCK_K
        Weight_ptrs += BLOCK_K * stride_wk

    # optional: save the activation inputs
    if META["SAVE_ACT_INPUTS"]:
        ActInputs_ptrs = ACT_INPUTS + rm[:, None] * stride_aim + rn[None, :]
        tl.store(ActInputs_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        acc = META["ACTIVATION"](acc)

    # write back result
    Out_ptrs = OUT + rm[:, None] * stride_dm + rn[None, :]
    tl.store(Out_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Activation needs to be a triton kernel
def fused_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation=None,
    save_inputs: bool = False
):
    """
    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    """

    if not x.is_contiguous():
        x = x.contiguous()

    x_ = x if x.ndim == 2 else x.flatten(0, 1)

    assert (
        x_.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions in between inputs and weight, {x_.shape} - {weight.shape}"
    assert bias is None or bias.is_contiguous()
    assert (
        bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"

    M, K = x_.shape
    N, K = weight.shape

    # FIXME: @lefaudeux
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_K"

    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_inputs else outputs

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(N, META["BLOCK_COL"]),
        )

    # fmt: off
    kernel_fma[grid](
        # data ptrs
        outputs, x_, weight,
        bias if bias is not None else x,  # auto skip bias if not present
        act_inputs,
        # shapes
        M, N, K,
        # strides
        outputs.stride(0), x_.stride(0),
        weight.stride(0), weight.stride(1),
        act_inputs.stride(0),
        # optional fused activation
        ACTIVATION=activation,
        # optional fused bias
        BIAS=bias is not None,
        # speed optimization: group the programs
        # improve on data reuse in L2 cache
        GROUP_ROW=8,
        BLOCK_K=32,
        SAVE_ACT_INPUTS=save_inputs
    )
    # fmt: on

    outputs = outputs if x.ndim == 3 else outputs.squeeze(0)

    return (outputs, act_inputs) if save_inputs else (outputs, None)


# fmt: off
@triton.autotune(
    configs=[
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 32 , 'BLOCK_COL': 64}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 256}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 256}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 32}, num_stages=4, num_warps=4),
        ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_grad_inputs(
    # Pointers to all the tensors
    GRAD_IN, GRAD_ACT, ACT_IN, GRAD_OUT, W,
    # Tensor dimensions
    M, N, K,
    # strides for all the gradients
    stride_gim, stride_gam, stride_gom,
    # strides for the extra data
    stride_aim, stride_wn, stride_wk,
    # Meta-parameters
    **META,
):
    # fmt: on
    """
    Kernel for computing `grad_out = grad_in * activation_grad(inputs) @ W^T`
    - grad_out has shape (M, N)
    - grad_act has shape (M, N)
    - W has shape (K, N)
    - grad_in has shape (M, K)
    - X has shape (M, K)

    This kernel will consolidate over N
    """
    # extract metaparameters
    BLOCK_M, GROUP_M = META["BLOCK_ROW"], META["GROUP_ROW"]
    BLOCK_N, BLOCK_K = META["BLOCK_N"], META["BLOCK_COL"]

    # programs are grouped together to improve L2 hit rate
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    pid_m = first_pid_m + (pid % GROUP_M)  # row-id of the program in the *launch grid*
    pid_k = (
        pid % num_pid_in_group
    ) // GROUP_M  # col-id of the program in the *launch grid*

    # memory ranges
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)

    # memory blocks can be computed using numpy-style broadcasting
    grad_out_ptrs = GRAD_OUT + rm[:, None] * stride_gom + rn[None, :]
    grad_act_ptrs = GRAD_ACT + rm[:, None] * stride_gam + rn[None, :]
    act_in_ptrs = ACT_IN + rm[:, None] * stride_aim + rn[None, :]
    weight_ptrs = W + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    # initialize and iteratively update accumulator
    grad_in = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    act_grad_fn = META["ACTIVATION_GRAD"]

    for _ in range(N, 0, -BLOCK_N):
        grad_out = tl.load(grad_out_ptrs)  # BLOCK_M x BLOCK_N
        grad_out_ptrs += BLOCK_N

        w = tl.load(weight_ptrs)  # BLOCK_N x BLOCK_K
        weight_ptrs += BLOCK_N * stride_wn

        # optional fused activation gradient (while the data is in shared memory)
        if META["ACTIVATION_GRAD"]:
            if META["ACTIVATION_GRAD_REQ_INPUTS"]:
                # This activation requires its inputs
                act_input = tl.load(act_in_ptrs)
                grad_act = act_grad_fn(act_input)
                act_in_ptrs += BLOCK_N
            else:
                # Save some time, we can reuse the outputs to know about the grad
                grad_act = act_grad_fn(grad_out)

            grad_out *= grad_act.to(grad_out.dtype)

        # store grad_act as an intermediate, will be used for grad/weight and grad/bias
        if META["SAVE_ACT_GRAD"]:
            tl.store(grad_act_ptrs, grad_out)
            grad_act_ptrs += BLOCK_N

        # gradient #1: input with respect to outputs
        # grad_in is grad_out scaled by the (transposed) weight
        grad_in += tl.dot(grad_out, w)

    # write back result
    grad_in_ptrs = GRAD_IN + rm[:, None] * stride_gim + rk[None, :]
    tl.store(grad_in_ptrs, grad_in, mask=(rm[:, None] < M) & (rk[None, :] < K))


# fmt: off
@triton.autotune(
    configs=[
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 32 , 'BLOCK_COL': 64}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 256}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_ROW': 256, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 256}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 64 , 'BLOCK_COL': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_COL': 32}, num_stages=4, num_warps=4),
        ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fused_grad_bias_grad_weight(
    # data ptrs
    G_BIAS, G_WEIGHT, G_ACT, INPUT,
    # shapes
    M, N, K,
    # strides
    stride_gam,
    stride_im,
    stride_gwn, stride_gwk,
    # Meta-parameters
    **META,
):
    # fmt: on

    """
    Compute the gradient with respect to the bias and weight

    - grad_bias has shape (N)
    - grad_act has shape (M, N)
    - grad_weight has shape (K, N)
    - inputs have shape (M, K)

    This kernel will walk along M:
        grad_bias <- sum(grad_act, M)
        grad_weight <- grad_act.transpose() @ inputs     ([N, K] = [N, M] x [M, K])
    """

    # extract metaparameters
    BLOCK_N, GROUP_N = META["BLOCK_ROW"], META["GROUP_ROW"]
    BLOCK_M, BLOCK_K = META["BLOCK_M"], META["BLOCK_COL"]

    # programs are grouped together to improve L2 hit rate
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_N * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_n = group_id * GROUP_N  # row-id of the first program in the group
    GROUP_N = min(
        num_pid_n - first_pid_n, GROUP_N
    )  # if `num_pid_n` isn't divisible by `GROUP_N`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    pid_n = first_pid_n + (pid % GROUP_N)  # row-id of the program in the *launch grid*
    pid_k = (
        pid % num_pid_in_group
    ) // GROUP_N  # col-id of the program in the *launch grid*

    # memory ranges
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)

    # memory blocks now
    grad_act_ptrs = G_ACT + rn[:, None] + rm[None, :] * stride_gam
    inputs_ptrs = INPUT + rm[:, None] * stride_im + rk[None, :]

    # initialize and iteratively update accumulator
    grad_weight_acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    grad_bias_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for _ in range(M, 0, -BLOCK_M):
        grad_act = tl.load(grad_act_ptrs, mask=(rn[:, None] < N) & (rm[None, :] < M), other=0.0)  # BLOCK_N x BLOCK_M
        inputs = tl.load(inputs_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)  # BLOCK_M x BLOCK_K

        grad_bias_acc += tl.sum(grad_act, axis=1).to(tl.float32)
        grad_weight_acc += tl.dot(grad_weight_acc, inputs).to(tl.float32)

        grad_act_ptrs += BLOCK_M * stride_gam
        inputs_ptrs += BLOCK_M * stride_im

    # write back results
    grad_weight_ptrs = G_WEIGHT + rn[:, None] * stride_gwn + rk[None, :] * stride_gwk
    tl.store(grad_weight_ptrs, grad_weight_acc, mask=(rn[:, None] < N) & (rk[None, :] < K))

    tl.store(G_BIAS + rn, grad_bias_acc, mask=rn < N)


# Activation needs to be a triton kernel
def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_inputs: Optional[torch.Tensor],
    activation_grad=None,
    activation_grad_req_inputs: bool = False,
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    """

    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)

    assert (
        grad_out_.shape[1] == weight.shape[0]
    ), "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, K = weight.shape

    grad_in = torch.empty((M, K), device=grad_out_.device, dtype=grad_out_.dtype)
    grad_act = torch.empty_like(grad_out_)

    # Compute the gradient for the inputs
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(K, META["BLOCK_COL"]),
        )

    if activation_inputs is None:
        # place holder, this will not be used really
        activation_inputs = grad_out_

    # fmt: off
    kernel_grad_inputs[grid](
        # data ptrs
        grad_in, grad_act, activation_inputs, grad_out_, weight,
        # shapes
        M, N, K,
        # strides
        grad_in.stride(0),
        grad_act.stride(0),
        grad_out_.stride(0),
        activation_inputs.stride(0),
        weight.stride(0), weight.stride(1),
        # optional fused activation
        ACTIVATION_GRAD=activation_grad,
        # data reuse optimization
        GROUP_ROW=8,
        BLOCK_N=32,
        ACTIVATION_GRAD_REQ_INPUTS=activation_grad_req_inputs,
        SAVE_ACT_GRAD=trainable_weight or trainable_bias
    )
    # fmt: on

    if trainable_bias and trainable_weight:
        # Fuse the bias and weight grad computations
        grad_bias: Optional[torch.Tensor] = torch.empty((N,), device=grad_out_.device, dtype=grad_out_.dtype)
        grad_weight : Optional[torch.Tensor] = torch.empty_like(weight)
        inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_ROW"]) * triton.cdiv(K, META["BLOCK_COL"]),
            )

        # fmt: off
        # This kernel will go over the M dimension
        kernel_fused_grad_bias_grad_weight[grid](
            # data ptrs
            grad_bias, grad_weight, grad_act, inputs_,
            # shapes
            M, N, K,
            # strides
            grad_act.stride(0), inputs_.stride(0),
            grad_weight.stride(0), grad_weight.stride(1),  # type: ignore   # mypy is drunk
            # data reuse optimization
            GROUP_ROW=8,
            BLOCK_M=32,
        )
        # fmt: on

    else:
        grad_bias = torch.sum(grad_act, dim=0) if trainable_bias else None

        # Reuse Triton optimized matmul
        grad_weight = None
        if trainable_weight:
            inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)
            grad_weight = triton.ops.matmul(grad_act.transpose(1, 0), inputs_)

    del grad_act

    return grad_in.reshape_as(inputs), grad_weight, grad_bias
