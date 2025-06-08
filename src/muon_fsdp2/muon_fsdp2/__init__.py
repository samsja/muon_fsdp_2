# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

from typing import Generator
from collections import deque

import torch
from torch.optim.optimizer import ParamsT
from torch.distributed.tensor import DTensor, Shard
from torch.distributed import gather, scatter
import torch.distributed as dist
from torch import Tensor

__version__ = "0.2.1"

__all__ = ["Muon", "MuonDDP"]


@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz(G, steps=10, eps=1e-7, f_iter=nsloop_torch):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
        # DTensor will NaN for sharded compute on Shard(1)
        if isinstance(X, DTensor):
            p = [Shard(0) if isinstance(p, Shard) else p for p in X._spec.placements]
            X = X.redistribute(placements=p)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)  # ensure top singular value <= 1
    X = f_iter(X, steps)
    return X if G.size(-2) <= G.size(-1) else X.mT


def paramst_to_groups(params: ParamsT) -> list[dict]:
    if all(isinstance(p, dict) for p in params):
        return params
    if all(isinstance(p, torch.nn.Parameter) for p in params):
        return [dict(params=params)]
    if all(isinstance(p, list) for p in params):
        return [dict(params=p) for p in params]
    raise ValueError(f"Invalid paramst_to_groups input: {params}")


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    This optimizer supports both Muon and Adam optimization through the use_muon flag.
    
    Some warnings:
    - Muon should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should use Adam (use_muon=False).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    This is a pruned implementation which uses the following hardcoded behaviors:
    * assumed used of 2D+ DTensor parameters, which will always be true if you use FSDP2.
    * nestrov momentum (on the input to NS)
    * EMA momentum (unlike the original Muon, which uses .mul_(momentum))

    Arguments:
        params: Params/groups to be optimized. Each group should have use_muon flag.

    For Muon groups (use_muon=True):
        lr: Learning rate.
        wd: Weight decay.
        momentum: momentum buffer decay rate.
        ns_steps: Newton-Schulz iteration steps.
        
    For Adam groups (use_muon=False):
        lr: Learning rate.
        betas: Adam beta parameters.
        eps: Adam epsilon.
        wd: Weight decay.
    """

    def __init__(
        self, params: ParamsT, *, lr: float | None = None, wd: float = 0.01, momentum: float = 0.95, ns_steps: int = 5
    ):
        # setup torch optimizer
        groups = paramst_to_groups(list(params))
        
        # Process groups to set defaults based on use_muon flag
        for group in groups:
            if "use_muon" not in group:
                group["use_muon"] = True  # default to Muon
                
            if group["use_muon"]:
                # Muon defaults
                group.setdefault("lr", lr or 0.02)
                group.setdefault("wd", wd)
                group.setdefault("momentum", momentum)
                group.setdefault("ns_steps", ns_steps)
            else:
                # Adam defaults
                group.setdefault("lr", lr or 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("wd", wd)
                
        super().__init__(groups, {})
        
        # init buffers ahead of time
        for pg in self.param_groups:
            for p in pg["params"]:
                assert isinstance(p, DTensor), "We only support DTensor. Use FSDP2."
                self.mesh = p._spec.device_mesh
                
                if pg["use_muon"]:
                    self.state[p] = dict(m=torch.zeros_like(p))
                    if p.ndim < 2:
                        raise ValueError(f"0/1D parameters are banned from Muon; user provided {p.shape=}")
                    if p.ndim > 2:
                        print(f"WARNING: muon used for {p.shape=}")
                else:
                    # Adam state
                    self.state[p] = dict(
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                        step=0
                    )

    def filter_group(self, group: dict) -> Generator[tuple[DTensor, DTensor, DTensor, int], None, None]:
        if group["use_muon"]:
            pg, lr, wd, momentum = group["params"], group["lr"], group["wd"], group["momentum"]
            pg = [p for p in pg if p.grad is not None]
            list_p = [p.data for p in pg]
            list_g = [p.grad.flatten(1) for p in pg]
            list_m = [self.state[p]["m"] for p in pg]
            torch._foreach_lerp_(list_m, list_g, 1 - momentum)  # EMA momentum
            torch._foreach_lerp_(list_g, list_m, momentum)  # nestrov momentum (for NS input)
            # Note: weight decay moved to deferred_work after NS
            yield from zip(list_p, list_g, list_m)

    @torch.no_grad()
    def step(self, *, prefetch_factor: int = 8):  # <-- changeme to 1 if you have numerical bugs
        # Handle Adam parameters first (simpler, no distributed ops)
        for group in self.param_groups:
            if not group["use_muon"]:
                lr, wd = group["lr"], group["wd"]
                betas, eps = group["betas"], group["eps"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        
                    state = self.state[p]
                    state["step"] += 1
                    
                    # Adam update
                    update = adam_update(
                        p.grad, 
                        state["exp_avg"], 
                        state["exp_avg_sq"],
                        state["step"], 
                        betas, 
                        eps
                    )
                    
                    # Apply weight decay and update
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        
        # Handle Muon parameters with distributed Newton-Schulz
        muon_groups = [g for g in self.param_groups if g["use_muon"]]
        if not muon_groups:
            return
            
        # fsdp sharding mesh dim is always last
        r, ws = self.mesh.get_local_rank(-1), self.mesh.size(-1)

        dq = deque()

        def deferred_work(p, g, g_full_block, spec, lr, wd, src_rank, rank):
            if rank == src_rank:
                chunks = list(g_full_block.chunk(ws, dim=0))
                scatter(g.to_local(), chunks, src=src_rank, async_op=True)
            else:
                scatter(g.to_local(), None, src=src_rank, async_op=True) 

       
            # Apply weight decay after NS (matching reference implementation)
            p.mul_(1 - lr * wd)
            # update parameter with NS'd grad
            lr_scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
            p.add_(g, alpha=-lr * lr_scale)

        i = 0
        for group in muon_groups:
            for p, g, m in self.filter_group(group):
                spec = g._spec
                dest_rank = i  % ws
                if dest_rank == r:
                    gather_lists = [torch.zeros_like(g.to_local()) for _ in range(ws)]
                    gather(g.to_local(), gather_lists, dst=dest_rank, async_op=True) 
                    g_full_block = torch.cat(gather_lists, dim=0)
                    g_full_block.copy_(zeropower_via_newtonschulz(g_full_block, steps=group["ns_steps"]))
                    g_full_block = g_full_block.view_as(p).type_as(p)
                else:
                    
                    g_local = g.to_local()
                    gather(g_local, None, dst=dest_rank, async_op=True)
                    g_full_block = None
                    
                dq.append([p, g, g_full_block, spec, group["lr"], group["wd"], dest_rank, r])
                if len(dq) > prefetch_factor:
                    deferred_work(*dq.popleft())
                i += 1
        for ls in dq:
            deferred_work(*ls)




def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class MuonDDP(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    This optimizer supports both Muon and Adam optimization through the use_muon flag.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        wd: Weight decay.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, wd=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        
        # Handle different param formats
        if isinstance(params, list) and len(params) > 0:
            if isinstance(params[0], dict):
                # Already param groups
                param_groups = params
            else:
                # List of parameters, create default group
                param_groups = [dict(params=params, use_muon=True)]
        else:
            raise ValueError("params must be a list of parameters or parameter groups")
            
        # Process groups and set defaults based on use_muon
        processed_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                group["use_muon"] = True
                
            if group["use_muon"]:
                # Muon defaults
                group["lr"] = group.get("lr", lr)
                group["wd"] = group.get("wd", wd)
                group["momentum"] = group.get("momentum", momentum)
                group["nesterov"] = group.get("nesterov", nesterov)
                group["ns_steps"] = group.get("ns_steps", ns_steps)
                
                # Sort muon params by size for efficient distribution
                group["params"] = sorted(list(group["params"]), key=lambda x: x.numel(), reverse=True)
                
                # Create update buffers for each unique size in muon params
                size_to_params = {}
                for p in group["params"]:
                    size = p.numel()
                    if size not in size_to_params:
                        size_to_params[size] = []
                    size_to_params[size].append(p)
                
                group["size_groups"] = []
                for size, params_list in size_to_params.items():
                    b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
                    size_group = {
                        "params": params_list,
                        "update_buffer": b,
                        "update_buffer_views": [b[i] for i in range(world_size)]
                    }
                    group["size_groups"].append(size_group)
            else:
                # Adam defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["wd"] = group.get("wd", wd)
                
            processed_groups.append(group)
            
        super().__init__(processed_groups, {})

    @torch.no_grad()
    def step(self):
        # Handle Adam parameters first
        for group in self.param_groups:
            if not group["use_muon"]:
                lr = group["lr"]
                wd = group["wd"]
                betas = group["betas"]
                eps = group["eps"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p.grad)
                        state["exp_avg_sq"] = torch.zeros_like(p.grad)
                        state["step"] = 0
                        
                    state["step"] += 1
                    
                    # Adam update
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        betas,
                        eps
                    )
                    
                    # Apply weight decay and update
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        
        # Handle Muon parameters with distributed processing
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
                
            # Process each size group separately
            for size_group in group["size_groups"]:
                update_buffer: Tensor = size_group["update_buffer"]
                update_buffer_views: list[Tensor] = size_group["update_buffer_views"]
                params: list[Tensor] = size_group["params"]
                
                handle = None
                params_world = None
                def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        # Apply weight decay after NS (matching reference implementation)
                        p_world.mul_(1 - group["lr"] * group["wd"])
                        p_world.add_(g_world.view_as(p_world),
                                     alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
                for base_i in range(len(params))[::self.world_size]:
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad
                        assert g is not None
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        if g.ndim == 4: # for the case of conv filters
                            g = g.view(len(g), -1)
                        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                    else:
                        g = update_buffer_views[self.rank]
                    if base_i > 0:
                        update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i : base_i + self.world_size]
                if params:  # Only call update_prev if we have params
                    update_prev()
