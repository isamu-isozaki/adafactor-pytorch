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
    eps1,
    eps2,
    scale_parameter,
    clip_threshold,
    n_elements,
    n_row_elements,
    n_column_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    row_mask = offsets < n_row_elements
    column_mask = offsets < n_column_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    offset_exp_avg_squared_row_ptr = exp_avg_squared_row_ptr + offsets
    offset_exp_avg_squared_column_ptr = exp_avg_squared_column_ptr + offsets

    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
    exp_avg_squared_row = tl.load(offset_exp_avg_squared_row_ptr, mask=row_mask)
    exp_avg_squared_column = tl.load(offset_exp_avg_squared_column_ptr, mask=column_mask)

    # Update lr
    # TODO: implement below. Currently, I'm not sure how to do this with dynamic shapes for p
    # if scale_parameter:
    #     param_rms = tl.sqrt(tl.sum(p*p)/n_elements)
    #     param_scale = tl.max(eps2, param_rms)
    # lr = param_scale * lr
    # stepweight decay

    p = p * (1 - lr * weight_decay)

    # get gradient squared

    update = grad*grad + eps1

    # update row and columns
    exp_avg_squared_row = exp_avg_squared_row*beta2 + tl.sum(update, axis=-1) * (1-beta2)
    exp_avg_squared_column = exp_avg_squared_column*beta2 + tl.sum(update, axis=-2) * (1-beta2)

    # approximate gradient

    r_factor = 1.0/tl.sqrt(exp_avg_squared_row / tl.sum(exp_avg_squared_row, axis=-1))[..., None]
    c_factor = 1.0/tl.sqrt(exp_avg_squared_column)[None]
    update = tl.dot(r_factor, c_factor)
    update = tl.dot(update, grad)
    denom = tl.sqrt(tl.sum(update*update)/n_elements)/ clip_threshold

    # clamp so the minimum is 1
    denom =  tl.where(denom < 1.0, 1.0, denom)
    update = update / denom

    # update momentum running average

    exp_avg = exp_avg*beta1 + update*(1-beta1)
    update = exp_avg

    p = p - lr*update


    # store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)
    tl.store(offset_exp_avg_squared_row_ptr, exp_avg_squared_row, mask = row_mask)
    tl.store(offset_exp_avg_squared_column_ptr, exp_avg_squared_column, mask = column_mask)



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
    eps1,
    eps2,
    clip_threshold,
    scale_parameter,
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
    offset_exp_avg_squared_ptr = exp_avg_squared_ptr + offsets

    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
    exp_avg_squared_ptr = tl.load(offset_exp_avg_squared_ptr, mask=mask)

    # Update lr
    # TODO: implement below. Currently, I'm not sure how to do this with dynamic shapes for p
    # if scale_parameter:
    #     param_rms = tl.sqrt(tl.sum(p*p)/n_elements)
    #     param_scale = tl.max(eps2, param_rms)
    # lr = param_scale * lr
    # stepweight decay

    p = p * (1 - lr * weight_decay)

    # get gradient squared

    update = grad*grad+ eps1

    # update exp_avg_squared
    exp_avg_squared = exp_avg_squared*beta2 + tl.sum(update, axis=-1) * (1-beta2)

    # approximate gradient

    update = grad / tl.sqrt(exp_avg_squared)

    # update momentum running average

    exp_avg = exp_avg*beta1 + update*(1-beta1)
    update = exp_avg

    p = p - lr*update


    # store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)
    tl.store(offset_exp_avg_squared_ptr, exp_avg_squared, mask = mask)
def update_fn(p, grad, exp_avg, lr, weight_decay, beta1, beta2, eps1, eps2, clip_threshold, factored=True, exp_avg_squared_row=None, exp_avg_squared_column=None, exp_avg_squared=None, scale_parameter=True):
    assert all([t.is_cuda for t in (p, grad)])
    # For now, we assume that exp_avg is not None.
    assert exp_avg is not None, "Assuming beta1 is not None"
    if exp_avg is not None:
        assert exp_avg.is_cuda
    if exp_avg_squared_row is not None:
        assert exp_avg_squared_row.is_cuda
        assert exp_avg_squared_column.is_cuda
        n_row_elements = exp_avg_squared_row.numel()
        n_column_elements = exp_avg_squared_column.numel()
    if exp_avg_squared is not None:
        assert exp_avg_squared.is_cuda

    n_elements = p.numel()
    print(p.shape)

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
            eps2,
            scale_parameter,
            clip_threshold,
            n_elements,
            n_row_elements,
            n_column_elements
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
            eps2,
            clip_threshold,
            scale_parameter,
            n_elements
        )
