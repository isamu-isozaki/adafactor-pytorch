import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()

# clone param and exp_avg before autotuning takes place
# as those are updated in-place

def clone_inplace_updated_params(nargs):
    for key in ['p_ptr', 'exp_avg_ptr', 'exp_avg_squared_row_ptr', 'exp_avg_squared_column_ptr', 'exp_avg_squared_ptr']:
        if nargs.get(key, None) is not None:
            nargs[key] = nargs[key].clone()

# triton cuda kernel

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4, pre_hook = clone_inplace_updated_params),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8, pre_hook = clone_inplace_updated_params),
], key = ['n_elements'])
@triton.jit
def matrix_update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_squared_row_ptr,
    exp_avg_squared_column_ptr,
    lr,
    weight_decay,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    exp_avg_squared_row_ptr = exp_avg_squared_row_ptr + offsets
    exp_avg_squared_column_ptr = exp_avg_squared_column_ptr + offsets

    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)

    # stepweight decay

    p = p * (1 - lr * weight_decay)

    # diff between momentum running average and grad

    diff = exp_avg - grad

    # weight update

    update = diff * beta1 + grad

    # torch.sign

    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # decay the momentum running average coefficient

    exp_avg = diff * beta2 + grad

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4, pre_hook = clone_inplace_updated_params),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8, pre_hook = clone_inplace_updated_params),
], key = ['n_elements'])
@triton.jit
def vector_update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_squared_ptr,
    lr,
    weight_decay,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    exp_avg_squared_ptr = exp_avg_squared_ptr + offsets
    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)

    # stepweight decay

    p = p * (1 - lr * weight_decay)

    # diff between momentum running average and grad

    diff = exp_avg - grad

    # weight update

    update = diff * beta1 + grad

    # torch.sign

    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # decay the momentum running average coefficient

    exp_avg = diff * beta2 + grad

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)

def update_fn(p, grad, exp_avg, lr, weight_decay, beta1, beta2, eps1, clip_threshold, factored=True, exp_avg_squared_row=None, exp_avg_squared_column=None, exp_avg_squared=None):
    assert all([t.is_cuda for t in (p, grad)])
    # For now, we assume that exp_avg is not None.
    assert exp_avg is not None, "Assuming beta1 is not None"
    if exp_avg is not None:
        assert exp_avg.is_cuda
    if exp_avg_squared_row is not None:
        assert exp_avg_squared_row.is_cuda
        assert exp_avg_squared_column.is_cuda
    if exp_avg_squared is not None:
        assert exp_avg_squared.is_cuda

    n_elements = p.numel()


    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    if factored:
        matrix_update_fn_kernel[grid](
            p,
            grad,
            exp_avg,
            exp_avg_squared_row,
            exp_avg_squared_column,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps1,
            clip_threshold,
            n_elements
        )
    else:
        vector_update_fn_kernel[grid](
            p,
            grad,
            exp_avg,
            exp_avg_squared,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps1,
            clip_threshold,
            n_elements
        )
