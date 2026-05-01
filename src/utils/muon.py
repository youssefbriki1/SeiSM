import torch
from torch.optim.optimizer import Optimizer
import math

class Muon(Optimizer):
    """
    Implements the Muon algorithm.
    An optimizer for 2D parameters of neural network hidden layers.
    """
    def __init__(self, params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True,
                 ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if adjust_lr_fn not in [None, "original", "match_rms_adamw"]:
            raise ValueError(f"Invalid adjust_lr_fn: {adjust_lr_fn}")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a tuple of exactly three floats: (a, b, c)")

        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum, 
            nesterov=nesterov,
            ns_coefficients=ns_coefficients, 
            eps=eps, 
            ns_steps=ns_steps,
            adjust_lr_fn="original" if adjust_lr_fn is None else adjust_lr_fn
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            a, b, c = group['ns_coefficients']
            eps = group['eps']
            ns_steps = group['ns_steps']
            adjust_lr_fn = group['adjust_lr_fn']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                if p.ndim != 2:
                    raise RuntimeError(
                        "Muon optimizer is designed for 2D parameters only. "
                        "Other parameters (like bias or embeddings) should use a standard method such as AdamW."
                    )
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    b_tilde = grad + momentum * buf
                else:
                    b_tilde = buf

                X = b_tilde
                
                transposed = X.size(0) > X.size(1)
                if transposed:
                    X = X.T
                
                X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
                
                for _ in range(ns_steps):
                    A = X @ X.T
                    B = b * A + c * (A @ A)
                    X = a * X + B @ X
                
                if transposed:
                    X = X.T
                
                O_t = X
                
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                A_dim, B_dim = p.shape
                if adjust_lr_fn == "original":
                    scale = math.sqrt(max(1.0, A_dim / B_dim))
                    adjusted_lr = lr * scale
                elif adjust_lr_fn == "match_rms_adamw":
                    scale = 0.2 * math.sqrt(max(A_dim, B_dim))
                    adjusted_lr = lr * scale
                else:
                    adjusted_lr = lr
                
                p.add_(O_t, alpha=-adjusted_lr)

        return loss