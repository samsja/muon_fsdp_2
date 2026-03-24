# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

import math
from typing import Protocol
import torch
from torch.distributed.tensor import DTensor
from torch.distributed import  gather, scatter
from collections import deque

__version__ = "0.3.0"

__all__ = ["Muon"]



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

def zeropower_via_newtonschulz5(G, steps: int):
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
    X = nsloop_torch(X, steps, a=a, b=b, c=c)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def apply_momentum(grad, momentum, beta, nesterov):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    return update

def apply_scaling(grad, rms_scale=False ):
    if rms_scale:
        # https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d05865fe90d9f39abcbcbd/examples/toy_train.py#L146
        grad *= 0.2 * math.sqrt(max(grad.shape[1], grad.shape[0]))
        return grad
    else:
        # https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L40
        grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return grad

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



class Work(Protocol):
    
    def __init__(self, param, state, group, index: int):
        ...
    
    def start(self):
        ...
    
    def finish(self):
        ...
    
    
class Fsdp1dWork:
    """
    muon handle for fsdp2 1d mesh.
    """
    
    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        
        self.index = index
    
        self._intermediate_state = None
    
    def start(self):

        self.param.grad = apply_momentum(self.param.grad, self.state["momentum_buffer"] , self.group["momentum"], self.group["nesterov"])
        
        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"
        
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        
        dest_rank = self.index % world_size
        
            
        if rank == dest_rank:
            gather_lists = [torch.zeros_like(input=grad.to_local()) for _ in range(world_size)]
            gather_handle = gather(grad.to_local(), gather_lists, group_dst=dest_rank, group=pg, async_op=True)
            
        else:
            gather_lists = None
            gather_handle = gather(grad.to_local(), None, group_dst=dest_rank, group=pg, async_op=True)
            
        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

    def finish(self):
        
        assert self._intermediate_state is not None, "gather work must be called first"
        
        grad = self.param.grad
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        
        dest_rank, gather_handle, gather_lists = self._intermediate_state
        gather_handle.wait()
        if rank == dest_rank:
            g_full_block = torch.cat(gather_lists, dim=0)
            g_full_block.copy_(zeropower_via_newtonschulz5(g_full_block, self.group["ns_steps"]))
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(grad.to_local(), scatter_list=chunks, src=dest_rank, group=pg, async_op=False)
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)
        
        update = apply_scaling(grad, self.group["rms_scale"])

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])


class Hsdp2dWork:
    """
    Muon work for HSDP (Hybrid Sharded Data Parallel) with 2D mesh.
    Assumes mesh dimension 0 is for data parallelism (replicas across nodes)
    and mesh dimension 1 is for FSDP (sharding within nodes).
    
    For example, with 2 nodes of 8 GPUs each:
    - mesh shape: (2, 8)
    - dim 0: replica group (2 nodes)
    - dim 1: FSDP group (8 GPUs per node)
    """
    
    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        self.index = index
        self._intermediate_state = None
    
    def start(self):
        self.param.grad = apply_momentum(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"]
        )
        
        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 2, "only supports 2D mesh for HSDP"
        
        device_mesh = grad.device_mesh
        
        # Get the FSDP dimension (typically dim 1)
        # Assuming the mesh is structured as (replicas, fsdp_shards)
        fsdp_dim = 1
        replica_dim = 0
        
        # Get mesh information
        mesh_coords = device_mesh.get_coordinate()
        if mesh_coords is None:
            raise RuntimeError("Could not get mesh coordinates")
        
        replica_rank = mesh_coords[replica_dim]
        fsdp_rank = mesh_coords[fsdp_dim]
        
        replica_size = device_mesh.size(replica_dim)
        fsdp_size = device_mesh.size(fsdp_dim)
        
        # Get process groups for each dimension
        fsdp_pg = device_mesh.get_group(fsdp_dim)
        replica_pg = device_mesh.get_group(replica_dim)
        
        # Determine which rank in the FSDP group will handle this parameter
        dest_fsdp_rank = self.index % fsdp_size
        
        # Step 1: Gather within FSDP group (along fsdp_dim)
        if fsdp_rank == dest_fsdp_rank:
            gather_lists = [torch.zeros_like(grad.to_local()) for _ in range(fsdp_size)]
            gather_handle = gather(
                grad.to_local(),
                gather_lists,
                group_dst=dest_fsdp_rank,
                group=fsdp_pg,
                async_op=True
            )
        else:
            gather_lists = None
            gather_handle = gather(
                grad.to_local(),
                None,
                group_dst=dest_fsdp_rank,
                group=fsdp_pg,
                async_op=True
            )
        
        self._intermediate_state = {
            'dest_fsdp_rank': dest_fsdp_rank,
            'fsdp_rank': fsdp_rank,
            'replica_rank': replica_rank,
            'fsdp_size': fsdp_size,
            'replica_size': replica_size,
            'fsdp_pg': fsdp_pg,
            'replica_pg': replica_pg,
            'gather_handle': gather_handle,
            'gather_lists': gather_lists,
            'fsdp_dim': fsdp_dim,
            'replica_dim': replica_dim
        }
    
    def finish(self):
        assert self._intermediate_state is not None, "start() must be called first"
        
        grad = self.param.grad
        state = self._intermediate_state
        
        # Wait for FSDP gather to complete
        state['gather_handle'].wait()
        
        # Step 2: Process on the designated rank within each replica
        if state['fsdp_rank'] == state['dest_fsdp_rank']:
            # Concatenate gathered gradients
            g_full_block = torch.cat(state['gather_lists'], dim=0)
            
            # Step 3: All-reduce across replicas (data parallel dimension)
            # Average the gradients across replicas
            if state['replica_size'] > 1:
                torch.distributed.all_reduce(
                    g_full_block,
                    op=torch.distributed.ReduceOp.AVG,
                    group=state['replica_pg']
                )
            
            # Step 4: Apply Newton-Schulz orthogonalization
            g_full_block.copy_(
                zeropower_via_newtonschulz5(g_full_block, self.group["ns_steps"])
            )
            g_full_block = g_full_block.type_as(grad)
            
            # Step 5: Scatter back within FSDP group
            chunks = list(g_full_block.chunk(chunks=state['fsdp_size'], dim=0))
            scatter(
                grad.to_local(),
                scatter_list=chunks,
                group_src=state['dest_fsdp_rank'],
                group=state['fsdp_pg'],
                async_op=False
            )
        else:
            # Other ranks in FSDP group just receive their chunk
            scatter(
                grad.to_local(),
                None,
                group_src=state['dest_fsdp_rank'],
                group=state['fsdp_pg'],
                async_op=False
            )
        
        # Step 6: Apply scaling and update parameters
        update = apply_scaling(grad, self.group["rms_scale"])
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])
        
        self._intermediate_state = None


class TpFsdp2dWork:
    """
    Muon work for TP + FSDP mesh (not yet implemented)
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("TP + FSDP not implemented, use HsdpFsdp2dWork for HSDP")
    
class EpFsdp2dWork:
    """
    Muon work for EP mesh
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")
    
class TpEpFsdp3dWork:
    """
    Muon work for TP + EP mesh
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")

class SingelDeviceWork:
    """
    muon handle for single device.
    """
    
    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        
    def start(self):
        update = muon_update(self.param.grad, self.state["momentum_buffer"], self.group["momentum"], self.group["nesterov"], self.group["ns_steps"], self.group["rms_scale"])
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])
        
    def finish(self):
        pass
    
    
class Muon(torch.optim.Optimizer):
    """
    DTensor variant of Muon, original code https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py
    also support single device variant.
    
    Notable changes:
        - add rms_scale argument to the optimizer following the moonlight paper https://arxiv.org/abs/2502.16982
    
    example usage:
    
    ```python
    
    from muon_fsdp2 import Muon


    optimizer = Muon([
        dict(
            params=model.square_params(),
            lr=1e-3,
            use_muon=True
        ),
        dict(
            params=model.non_square_params(),
            lr=1e-3,
            use_muon=False
        )
    ])   
    ```
    
    
    param_groups args:
        lr: learning rate
        momentum: momentum
        weight_decay: weight decay
        use_muon: whether to use muon
        rms_scale: whether to scale the gradient by the RMS of the gradient . If true use the rms scale from the moonlight paper.
                https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d1d7faa3c2cd45acba4666/examples/toy_train.py#L146
                This variant adjust the update so that the RMS match the one of adam, allowing to only have one learning rate for all parameters.

    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["rms_scale"] = group.get("rms_scale", True)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "rms_scale", "nesterov", "ns_steps"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    def _get_work_class(self, p: torch.Tensor) -> tuple[type[Work], int]:
        """
        dispatch the work class based on the mesh dimension.
        For 2D mesh, uses Hsdp2dWork which assumes:
        - dim 0: replica/data parallel dimension (across nodes)
        - dim 1: FSDP dimension (sharding within nodes)
        """
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return Hsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingelDeviceWork, 1
        
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        dq: deque[Work] = deque()

        for group in self.param_groups:
            
            if group["use_muon"]:
                for i ,p in enumerate(group["params"]):
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    
                    class_work, prefetch_factor = self._get_work_class(p)
                        
                    work = class_work(p, state, group, i)
                    work.start()
                    dq.append(work)
                    
                    
                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        for work in dq:
            work.finish()
            
        return loss
    


    
