"""Portable MambaVision operators for Ascend NPU.

MambaVision imports ``mamba_ssm.ops.selective_scan_interface`` even though its
pip dependency builds a CUDA extension.  This module provides the same scan
contract with ordinary PyTorch tensor operations and patches only the forward
methods whose upstream implementations select CUDA-oriented fused kernels.
All learnable modules and parameter names remain owned by MambaVision, so the
official pretrained checkpoint stays compatible.
"""

from __future__ import annotations

import sys
from types import MethodType, ModuleType
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MambaNPUPatchReport(NamedTuple):
    mixers: int
    attentions: int


def _compute_dtype(tensor):
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def _validate_scan_inputs(u, delta, A, B, C):
    if u.ndim != 3 or delta.ndim != 3:
        raise ValueError("u and delta must have shape [batch, channels, length]")
    if u.shape != delta.shape:
        raise ValueError(f"u/delta shape mismatch: {u.shape} vs {delta.shape}")
    batch, channels, length = u.shape
    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError(
            f"A must have shape [channels, state], got {tuple(A.shape)}"
        )
    state_size = A.shape[1]
    for name, value in (("B", B), ("C", C)):
        valid_static = value.ndim == 2 and value.shape == (channels, state_size)
        valid_variable = value.ndim == 3 and value.shape == (
            batch,
            state_size,
            length,
        )
        if not (valid_static or valid_variable):
            raise ValueError(
                f"{name} must be [channels,state] or [batch,state,length], "
                f"got {tuple(value.shape)}"
            )


def _expand_scan_parameter(value, batch, channels, state_size, length):
    if value.ndim == 2:
        return value.reshape(1, channels, state_size, 1)
    return value.reshape(batch, 1, state_size, length)


def _parallel_scan_output(u, delta, A, B, C):
    """Inclusive associative scan for ``state_t = a_t*state_(t-1)+b_t``."""
    batch, channels, length = u.shape
    state_size = A.shape[1]
    B = _expand_scan_parameter(B, batch, channels, state_size, length)
    C = _expand_scan_parameter(C, batch, channels, state_size, length)
    delta_expanded = delta.unsqueeze(2)
    transition = torch.exp(delta_expanded * A.reshape(1, channels, state_size, 1))
    state = delta_expanded * B * u.unsqueeze(2)

    # Function composition is associative:
    # (a2, b2) o (a1, b1) = (a2*a1, a2*b1+b2).
    # Hillis-Steele scan reduces Python/NPU launches from O(length) to O(log length).
    offset = 1
    while offset < length:
        right_transition = transition[..., offset:]
        right_state = state[..., offset:]
        composed_transition = right_transition * transition[..., :-offset]
        composed_state = right_state + right_transition * state[..., :-offset]
        transition = torch.cat(
            (transition[..., :offset], composed_transition), dim=-1
        )
        state = torch.cat((state[..., :offset], composed_state), dim=-1)
        offset *= 2

    output = (state * C).sum(dim=2)
    return output, state[..., -1]


def _sequential_scan_output(u, delta, A, B, C):
    """Small-memory reference path used for validation and constrained runs."""
    batch, channels, length = u.shape
    state_size = A.shape[1]
    B = _expand_scan_parameter(B, batch, channels, state_size, length)
    C = _expand_scan_parameter(C, batch, channels, state_size, length)
    state = u.new_zeros((batch, channels, state_size))
    outputs = []
    for index in range(length):
        step = delta[..., index].unsqueeze(-1)
        transition = torch.exp(step * A.unsqueeze(0))
        B_step = B[..., 0] if B.shape[-1] == 1 else B[..., index]
        C_step = C[..., 0] if C.shape[-1] == 1 else C[..., index]
        state = transition * state + step * B_step * u[..., index].unsqueeze(-1)
        outputs.append((state * C_step).sum(dim=-1))
    return torch.stack(outputs, dim=-1), state


def _checkpoint_scan(function, *inputs):
    if not torch.is_grad_enabled() or not any(value.requires_grad for value in inputs):
        return function(*inputs)
    try:
        return checkpoint(function, *inputs, use_reentrant=False)
    except TypeError:
        # torch_npu deployments paired with older PyTorch releases do not
        # expose the use_reentrant keyword.
        return checkpoint(function, *inputs)


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    *,
    backend="parallel",
    checkpoint_scan=True,
):
    """NPU-safe subset of ``mamba_ssm.selective_scan_fn``.

    The variable B/C layout emitted by MambaVision is fully supported. Static
    per-channel B/C tensors are also accepted for unit-level compatibility.
    """
    _validate_scan_inputs(u, delta, A, B, C)
    backend = str(backend).strip().lower()
    if backend not in {"parallel", "sequential"}:
        raise ValueError("Mamba NPU scan backend must be parallel or sequential")

    output_dtype = u.dtype
    compute_dtype = _compute_dtype(u)
    u_compute = u.to(dtype=compute_dtype)
    delta_compute = delta.to(dtype=compute_dtype)
    A_compute = A.to(dtype=compute_dtype)
    B_compute = B.to(dtype=compute_dtype)
    C_compute = C.to(dtype=compute_dtype)
    if delta_bias is not None:
        delta_compute = delta_compute + delta_bias.to(
            dtype=compute_dtype
        ).reshape(1, -1, 1)
    if delta_softplus:
        delta_compute = F.softplus(delta_compute)

    scan = (
        _parallel_scan_output if backend == "parallel" else _sequential_scan_output
    )
    if checkpoint_scan:
        output, last_state = _checkpoint_scan(
            scan, u_compute, delta_compute, A_compute, B_compute, C_compute
        )
    else:
        output, last_state = scan(
            u_compute, delta_compute, A_compute, B_compute, C_compute
        )

    if D is not None:
        output = output + u_compute * D.to(dtype=compute_dtype).reshape(1, -1, 1)
    if z is not None:
        output = output * F.silu(z.to(dtype=compute_dtype))
    output = output.to(dtype=output_dtype)
    if return_last_state:
        return output, last_state
    return output


def install_mamba_ssm_npu_shim(backend="parallel", checkpoint_scan=True):
    """Install the import surface required by the upstream MambaVision package."""
    backend = str(backend).strip().lower()
    if backend not in {"parallel", "sequential"}:
        raise ValueError("Mamba NPU scan backend must be parallel or sequential")
    root = ModuleType("mamba_ssm")
    root.__path__ = []
    root.__version__ = "toolrgs-npu-portable"
    ops = ModuleType("mamba_ssm.ops")
    ops.__path__ = []
    interface = ModuleType("mamba_ssm.ops.selective_scan_interface")

    def configured_scan(*args, **kwargs):
        kwargs.setdefault("backend", backend)
        kwargs.setdefault("checkpoint_scan", checkpoint_scan)
        return selective_scan_fn(*args, **kwargs)

    interface.selective_scan_fn = configured_scan
    root.ops = ops
    ops.selective_scan_interface = interface
    sys.modules["mamba_ssm"] = root
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = interface
    return interface


def _depthwise_conv1d_as_conv2d(input_tensor, convolution):
    kernel = int(convolution.weight.shape[-1])
    # PyTorch's stride-1 padding='same' puts the extra element on the right
    # for even kernels. MambaVision uses d_conv=4 by default, so preserving
    # this asymmetry is required to keep both sequence length and numerics.
    total_padding = kernel - 1
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    input_tensor = F.pad(
        input_tensor.unsqueeze(2),
        (left_padding, right_padding, 0, 0),
    )
    return F.conv2d(
        input_tensor,
        convolution.weight.unsqueeze(2),
        bias=convolution.bias,
        groups=convolution.groups,
    ).squeeze(2)


def _mambavision_mixer_forward_npu(self, hidden_states):
    batch, length, _channels = hidden_states.shape
    projected = self.in_proj(hidden_states).transpose(1, 2).contiguous()
    x, z = projected.chunk(2, dim=1)
    x = F.silu(_depthwise_conv1d_as_conv2d(x, self.conv1d_x))
    z = F.silu(_depthwise_conv1d_as_conv2d(z, self.conv1d_z))

    parameters = self.x_proj(x.transpose(1, 2).reshape(batch * length, -1))
    dt, B, C = torch.split(
        parameters,
        [self.dt_rank, self.d_state, self.d_state],
        dim=-1,
    )
    dt = self.dt_proj(dt).reshape(batch, length, -1).transpose(1, 2)
    B = B.reshape(batch, length, self.d_state).transpose(1, 2).contiguous()
    C = C.reshape(batch, length, self.d_state).transpose(1, 2).contiguous()
    y = selective_scan_fn(
        x,
        dt,
        -torch.exp(self.A_log.float()),
        B,
        C,
        self.D.float(),
        z=None,
        delta_bias=self.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=False,
        backend=self._toolrgs_scan_backend,
        checkpoint_scan=self._toolrgs_checkpoint_scan,
    )
    output = torch.cat((y, z), dim=1).transpose(1, 2).contiguous()
    return self.out_proj(output)


def patch_mambavision_for_npu(
    model,
    scan_backend="parallel",
    checkpoint_scan=True,
):
    """Patch CUDA-oriented MambaVision forwards without changing parameters."""
    scan_backend = str(scan_backend).strip().lower()
    if scan_backend not in {"parallel", "sequential"}:
        raise ValueError("Mamba NPU scan backend must be parallel or sequential")
    mixers = 0
    attentions = 0
    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name == "MambaVisionMixer":
            module._toolrgs_scan_backend = scan_backend
            module._toolrgs_checkpoint_scan = bool(checkpoint_scan)
            module.forward = MethodType(_mambavision_mixer_forward_npu, module)
            mixers += 1
        elif class_name == "Attention" and hasattr(module, "fused_attn"):
            # The upstream non-fused path is regular matmul/softmax and keeps
            # the exact same qkv/projection parameters.
            module.fused_attn = False
            attentions += 1
    if mixers == 0:
        raise RuntimeError(
            "No MambaVisionMixer modules were found; check mambavision==1.2.0"
        )
    return MambaNPUPatchReport(mixers=mixers, attentions=attentions)
