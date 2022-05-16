import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=True,
        fixed_decay=False,
        rectify=False,
        clip_step=None,
        norm_ord=2,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= norm_ord:
            raise ValueError("Invalid norm_ord value: {}".format(norm_ord))
        if rectify:
            raise NotImplementedError(
                f"AdaBelief does not support rectify={rectify}. use Fasle instead"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            rectify=rectify,
            clip_step=clip_step,
            norm_ord=norm_ord,
        )
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_group = []
        norm = 0
        for group in self.param_groups:
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            clip_step = group["clip_step"]
            eps = group["eps"]
            fixed_decay = group["fixed_decay"]
            lr = group["lr"]
            norm_ord = group["norm_ord"]
            rectify = group["rectify"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]

            steps = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients.")
                grad = p.grad.data
                state = self.state[p]

                # apply gradient
                if not weight_decouple and weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    keys = ["exp_avg", "exp_avg_var", "max_exp_avg_var"][: 2 + amsgrad]
                    try:
                        for I in keys:
                            state[I] = torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )
                    except:
                        for I in keys:
                            state[I] = torch.zeros_like(p.data)

                # get current state variable
                state["step"] += 1
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_var, max_exp_avg_var, out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_var.sqrt()
                else:
                    denom = exp_avg_var.sqrt()
                denom = denom.div_(bias_correction2**0.5).add_(eps)

                if clip_step is None:
                    # apply weight_decay
                    if weight_decouple and weight_decay != 0:
                        rate = [1, lr][fixed_decay] * weight_decay
                        p.data.mul_(1.0 - rate)
                    p.data.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                else:
                    step = (-lr / bias_correction1) * (exp_avg / denom)
                    # apply weight_decay
                    if weight_decouple and weight_decay != 0:
                        rate = [1, lr][fixed_decay] * weight_decay
                        step.add_(p.data, alpha=-rate)
                    norm += step.norm(p=norm_ord).cpu().numpy() ** norm_ord
                    steps.append(step)

            step_group.append(None if len(steps) == 0 else steps)

        if clip_step is None:
            norm = None
        else:
            norm **= 1 / norm_ord
            mult = min(1, clip_step / norm)
            for group, steps in zip(self.param_groups, step_group):
                if steps is None:
                    continue
                for p, step in zip(
                    (p for p in group["params"] if p.grad is not None), steps
                ):
                    p.add_(step, alpha=mult)

        return {"loss": loss, "norm": norm}
