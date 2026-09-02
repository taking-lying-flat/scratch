import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def fused_qk_norm_rope_cache_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    positions_ptr,
    slot_mapping_ptr,
    k_cache_ptr,
    v_cache_ptr,
    EPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM
    half_dim = HEAD_DIM // 2

    position = tl.load(positions_ptr + token_idx)
    rope_offsets = offsets % half_dim
    cos = tl.load(cos_sin_ptr + position * cos_sin_stride + rope_offsets, mask=mask, other=0.0)
    sin = tl.load(
        cos_sin_ptr + position * cos_sin_stride + half_dim + rope_offsets,
        mask=mask,
        other=0.0,
    )
    paired_offsets = tl.where(offsets < half_dim, offsets + half_dim, offsets - half_dim)
    rope_sign = tl.where(offsets < half_dim, -1.0, 1.0)

    if head_idx < NUM_Q_HEADS:
        q_offsets = token_idx * q_stride + head_idx * HEAD_DIM + offsets
        q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0).to(tl.float32)
        q_pair = tl.load(
            q_ptr + token_idx * q_stride + head_idx * HEAD_DIM + paired_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(q * q, axis=0) / HEAD_DIM + EPS)
        q_weight = tl.load(q_norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        q_pair_weight = tl.load(q_norm_weight_ptr + paired_offsets, mask=mask, other=0.0).to(tl.float32)
        q = (q * inv_rms * q_weight).to(q_ptr.dtype.element_ty)
        q_pair = (q_pair * inv_rms * q_pair_weight).to(q_ptr.dtype.element_ty)
        q_rot = q * cos + rope_sign * q_pair * sin
        tl.store(q_ptr + q_offsets, q_rot, mask=mask)
    else:
        kv_head_idx = head_idx - NUM_Q_HEADS
        k_offsets = token_idx * k_stride + kv_head_idx * HEAD_DIM + offsets
        k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0).to(tl.float32)
        k_pair = tl.load(
            k_ptr + token_idx * k_stride + kv_head_idx * HEAD_DIM + paired_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) / HEAD_DIM + EPS)
        k_weight = tl.load(k_norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        k_pair_weight = tl.load(k_norm_weight_ptr + paired_offsets, mask=mask, other=0.0).to(tl.float32)
        k = (k * inv_rms * k_weight).to(k_ptr.dtype.element_ty)
        k_pair = (k_pair * inv_rms * k_pair_weight).to(k_ptr.dtype.element_ty)
        k_rot = k * cos + rope_sign * k_pair * sin

        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot != -1:
            cache_offsets = slot * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM + offsets
            v = tl.load(
                v_ptr + token_idx * v_stride + kv_head_idx * HEAD_DIM + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(k_cache_ptr + cache_offsets, k_rot, mask=mask)
            tl.store(v_cache_ptr + cache_offsets, v, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def fused_qk_norm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype == k_cache.dtype == v_cache.dtype
    assert k.shape == v.shape == (num_tokens, num_kv_heads, head_dim)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert q.stride(1) == k.stride(1) == v.stride(1) == head_dim
    assert q_norm_weight.numel() == k_norm_weight.numel() == head_dim
    assert positions.numel() == slot_mapping.numel() == num_tokens
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_cache.stride(1) == v_cache.stride(1) == num_kv_heads * head_dim
    assert head_dim % 2 == 0 and head_dim <= 256
    block_d = triton.next_power_of_2(head_dim)
    fused_qk_norm_rope_cache_kernel[(num_tokens, num_q_heads + num_kv_heads)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        v,
        v.stride(0),
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        positions,
        slot_mapping,
        k_cache,
        v_cache,
        EPS=eps,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
    return q


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def can_fuse_decode(self, q: torch.Tensor) -> bool:
        context = get_context()
        return (
            not context.is_prefill
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim % 2 == 0
            and self.head_dim <= 256
            and self.k_cache.numel() > 0
            and self.v_cache.numel() > 0
        )

    def fused_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        context = get_context()
        q = fused_qk_norm_rope_cache(
            q,
            k,
            v,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            self.k_cache,
            self.v_cache,
            context.slot_mapping,
            eps,
        )
        return self.decode(q)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        context = get_context()
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            self.k_cache,
            self.v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = self.decode(q)
        return o
