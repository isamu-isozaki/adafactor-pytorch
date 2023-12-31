from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
import math

# functions

def exists(val):
    return val is not None

def rms(tensor):
    return (tensor.norm(2) / tensor.numel()) ** 0.5
def approx_gradient(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (
        (exp_avg_sq_row / exp_avg_sq_row.sum(dim=-1, keepdim=True))
        .rsqrt_()
        .unsqueeze(-1)
    )
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)
# update functions
def get_lr(lr, group, state):
    # update lr
    if group['relative_step']:
        min_step = (
            1e-6 * state["step"] if group["warmup_init"] else 1e-2
        )
        lr = min(min_step, 1.0 / math.sqrt(state["step"]))
    return lr
def update_fn(p, grad, exp_avg, lr, weight_decay, beta1, beta2, eps1, eps2, clip_threshold, factored=True, exp_avg_squared_row=None, exp_avg_squared_column=None, exp_avg_squared=None, scale_parameter=True):
    # update lr
    param_scale = 1
    if scale_parameter:
        param_rms = rms(p)
        param_scale = max(eps2, param_rms)
    lr = param_scale * lr
    if weight_decay != 0:
        p.add_(
            p, alpha=-weight_decay*lr
        )
    update = (grad**2) + eps1
    if factored:
        exp_avg_squared_row.mul_(beta2).add_(
            update.sum(dim=-1), alpha=1.0 - beta2
        )
        exp_avg_squared_column.mul_(beta2).add_(
            update.sum(dim=-2), alpha=1.0 - beta2
        )

        # Approximation of exponential moving average of square of gradient
        update = approx_gradient(exp_avg_squared_row, exp_avg_squared_column)
        update.mul_(grad)
    else:
        exp_avg_squared.mul_(beta2).add_(update, alpha=1.0 - beta2)
        update = exp_avg_squared.rsqrt().mul_(grad)

    update.div_(
        (rms(update) / clip_threshold).clamp_(min=1.0)
    )

    if beta1 is not None:
        exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
        update = exp_avg


    p.add_(update, alpha=-lr)

    if p.data.dtype in {torch.float16, torch.bfloat16}:
        p.data.copy_(p)


# class

class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        eps: float = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        betas: float = (None, None),
        weight_decay: float = 0.0,
        scale_parameter: bool = False,
        relative_step: bool = False,
        warmup_init: bool = False,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert decay_rate < 0
        defaults = dict(
            lr = lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            betas = betas,
            weight_decay = weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init
        )
        # This function gives beta for squared gradient
        self.decay_func = lambda t: 1-t**decay_rate

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from adafactor_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn
    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, weight_decay, betas, eps, clip_threshold, state = p.grad, group['lr'], group['weight_decay'], group['betas'], group['eps'], group['clip_threshold'], self.state[p]
                do_factor = len(grad.shape) >= 2
                # init state - exponential moving average of gradient values
                beta1 = betas[0]
                if len(state) == 0:
                    state['step'] = 0
                    if beta1 is not None:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if do_factor:
                        # Here, we are expecting row matrix x column matrix gives a good approximation of the exponential average of the squared gradient
                        state['exp_avg_squared_row'] = torch.zeros(grad.shape[:-1]).to(grad)
                        state['exp_avg_squared_column'] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)
                    else:
                        # If there's not enough dimensions to factor, just use the tensor shape
                        state['exp_avg_squared'] = torch.zeros_like(grad)

                else:
                    if beta1 is not None:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if do_factor:
                        state["exp_avg_squared_row"] = state["exp_avg_squared_row"].to(grad)
                        state["exp_avg_squared_column"] = state["exp_avg_squared_column"].to(grad)
                    else:
                        state["exp_avg_squared"] = state["exp_avg_squared"].to(grad)

                    exp_avg = state.get('exp_avg', None)
                    state["step"] += 1
                    if betas[1] is None:
                        beta2 = self.decay_func(state["step"])
                    else:
                        beta2 = betas[1]
                    lr = get_lr(lr, group, state)
                    self.update_fn(
                        p,
                        grad,
                        exp_avg,
                        lr,
                        weight_decay,
                        beta1,
                        beta2,
                        eps[0],
                        eps[1],
                        clip_threshold,
                        factored=do_factor,
                        exp_avg_squared_row=state.get('exp_avg_squared_row', None),
                        exp_avg_squared_column=state.get('exp_avg_squared_column', None),
                        exp_avg_squared=state.get('exp_avg_squared', None),
                        scale_parameter=group['scale_parameter']
                    )

        return loss
