from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
import math

# functions

def exists(val):
    return val is not None

def rms(tensor):
    return tensor.norm(2) / (tensor.numel() ** 0.5)
def approx_gradient(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)
# update functions

def update_fn(p, grad, exp_avg, lr, weight_decay, beta1, beta2, group, state):
    # update lr
    factored = len(grad.shape) >= 2
    if group['relative_step']:
        min_step = (
            1e-6 * state["step"] if group["warmup_init"] else 1e-2
        )
        lr = min(min_step, 1.0 / math.sqrt(state["step"]))
    param_scale = 1
    if group["scale_parameter"]:
        param_rms = rms(p)
        param_scale = max(group["eps"][1], param_rms)
    lr = param_scale * lr

    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_squared_row = state["exp_avg_squared_row"]
        exp_avg_squared_column = state["exp_avg_squared_column"]

        exp_avg_squared_row.mul_(beta2).add_(
            update.mean(dim=-1), alpha=1.0 - beta2
        )
        exp_avg_squared_column.mul_(beta2).add_(
            update.mean(dim=-2), alpha=1.0 - beta2
        )

        # Approximation of exponential moving average of square of gradient
        update = approx_gradient(exp_avg_squared_row, exp_avg_squared_column)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_(
        (rms(update) / group["clip_threshold"]).clamp_(min=1.0)
    )
    update.mul_(group["lr"])

    if group['beta1'] is not None:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
        update = exp_avg

    if group["weight_decay"] != 0:
        p.add_(
            p, alpha=-weight_decay*lr
        )

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
        beta1: float = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        use_triton: bool = False
    ):
        """Implements Adafactor algorithm.

        Taken from fairseq package from Meta and lion-pytorch. Thanks to both!

        This implementation is based on:
        `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
        (see https://arxiv.org/abs/1804.04235)

        Note that this optimizer internally adjusts the learning rate
        depending on the *scale_parameter*, *relative_step* and
        *warmup_init* options. To use a manual (external) learning rate
        schedule you should set `scale_parameter=False` and
        `relative_step=False`.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): external learning rate (default: 1e-4)
            eps (tuple[float, float]): regularization constans for square gradient
                and parameter scale respectively (default: (1e-30, 1e-3))
            clip_threshold (float): threshold of root mean square of
                final gradient update. This enforces max strength of update as opposed to gradient clipping.
                (default: 1.0)
            decay_rate (float): coefficient used to compute running averages of square
                gradient. This value is used to calculate Beta 2 which is responsible for
                the second moment update(square gradient) (default: -0.8)
            beta1 (float): coefficient used for computing running averages of gradient
                (default: None)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            scale_parameter (bool): if True, learning rate is scaled by root mean square of
                parameter (default: True)
            relative_step (bool): if True, time-dependent learning rate is computed
                instead of external learning rate. The rate is calculated as 1-1/sqrt(timesteps).
                The effect is the same as learning rate decay.
                (default: True)
            warmup_init (bool): time-dependent learning rate computation depends on
                whether warm-up initialization is being used. This is equivalent to warmup in lr schedulers.
                (default: False)
        """
        assert lr > 0.
        assert 0. <= beta1 <= 1
        assert decay_rate < 0

        defaults = dict(
            lr = lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1 = beta1,
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

                grad, lr, weight_decay, beta1, state = p.grad, group['lr'], group['weight_decay'], group['beta1'], self.state[p]
                do_factor = len(grad.shape) >= 2
                # init state - exponential moving average of gradient values

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

                    exp_avg = state['exp_avg']
                    state["step"] += 1
                    beta2 = self.decay_func(state["step"])
                    self.update_fn(
                        p,
                        grad,
                        exp_avg,
                        lr,
                        weight_decay,
                        beta1,
                        beta2,
                        group
                    )

        return loss
