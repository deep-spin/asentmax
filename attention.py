# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import functools
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
)

from torchtitan.tools.utils import has_cuda_capability

# AdaSplash kernels
from torchtitan.models.kernels.adaprefill_left_pad import sparse_attention_prefill
from torchtitan.models.kernels.adasplash15 import sparse_attn as adasplash15
from torchtitan.models.kernels.triton_entmax import triton_entmax_attention
from torchtitan.models.attention_bias import get_attn_mask_for_sdpa, get_slopes


# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]
FLEX_ATTN_SCORE_T = str


class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        mode="max-autotune-no-cudagraphs",
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}

    used_score_mod_types: ClassVar[set[FLEX_ATTN_SCORE_T]] = set()
    score_mods: ClassVar[dict[FLEX_ATTN_SCORE_T, Callable]] = {}

    # Instance variables.
    attn_mask_type: str
    score_mod_type: str

    def __init__(
        self,
        attn_mask_type: str,
        score_mod_type: str | None = None,
        fixed_block_size: int | None = None,
    ) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        if score_mod_type is not None and score_mod_type not in [
            "alibi",
            "alibi_with_nope",
            "linear_alibi_with_nope",
        ]:
            raise ValueError(f"Unrecognized score_mod_type {score_mod_type}.")

        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size

        FlexAttention.used_attn_mask_types.add(self.mask_key)

        self.score_mod_type = score_mod_type if score_mod_type is not None else "noop"
        FlexAttention.used_score_mod_types.add(self.score_mod_type)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        offset: torch.Tensor | None = None,  # used for decoding
    ) -> torch.Tensor:
        block_mask = FlexAttention.block_masks[self.mask_key]

        # FIXME compatibility with other classes
        # attention_mask is init in the main training loop now

        if self.score_mod_type == "noop" and "noop" not in FlexAttention.score_mods:
            FlexAttention.build_score_mod(None, None)

        score_mod = FlexAttention.score_mods[self.score_mod_type]
        enable_gqa = q.shape[1] != k.shape[1]

        if offset is not None:
            block_mask = None
            left_pad = torch.amax(offset, dim=0) - offset
            is_prefill = q.shape[2] != 1

            if self.score_mod_type in [
                "alibi",
                "alibi_with_nope",
                "linear_alibi",
                "linear_alibi_with_nope",
            ]:
                slopes = get_slopes(
                    num_heads=q.shape[1], method=self.score_mod_type
                ).to(q.device, q.dtype)
            else:
                slopes = torch.zeros(q.shape[1], dtype=q.dtype, device=q.device)

            if is_prefill:
                score_mod = FlexAttention.get_causal_score_mod_w_prefill_and_leftpad(
                    score_mod, left_pad, slopes=slopes
                )
            else:
                score_mod = FlexAttention.get_causal_score_mod_w_offset_and_leftpad(
                    score_mod, offset, left_pad, slopes=slopes
                )

        return FlexAttention.flex_attn(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=enable_gqa,
            scale=scale,
        )

    @staticmethod
    def get_causal_score_mod_w_prefill_and_leftpad(
        score_mod: _score_mod_signature,
        _left_pad: torch.Tensor,  # [B]  number of left-pad tokens per item
        slopes: torch.Tensor,
    ):
        def _score_mod(score, b, h, q, kv):
            lp = _left_pad[b]
            allow = (kv >= lp) & (q >= kv)

            return torch.where(
                allow,
                # score_mod(score, b, h, q, kv),
                score + (slopes[h] * (kv - q)),
                torch.tensor(-float("inf"), dtype=score.dtype, device=score.device),
            )

        return _score_mod

    @staticmethod
    def get_causal_score_mod_w_offset_and_leftpad(
        score_mod: _score_mod_signature,
        _offset: torch.Tensor,  # [B]
        _left_pad: torch.Tensor,  # [B]  number of left-pad tokens per item
        slopes: torch.Tensor,
    ):
        def _score_mod(score, b, h, q, kv):
            lp = _left_pad[b]
            off = _offset[b]
            q_abs = q + off
            allow = kv >= lp

            return torch.where(
                allow,
                # score_mod(score, b, h, q_abs, kv),
                score + (slopes[h] * (kv - q_abs)),
                torch.tensor(-float("inf"), dtype=score.dtype, device=score.device),
            )

        return _score_mod

    @staticmethod
    def _get_noop_score_mod():
        def noop_score_mod(score, b, h, q, kv):
            return score

        return noop_score_mod

    @staticmethod
    def _get_alibi_score_mod(heads, method, device):
        # from https://pytorch.org/blog/flexattention/

        slopes = get_slopes(heads, method).to(device)

        def alibi_mask(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ):
            bias = slopes[h] * (kv_idx - q_idx)
            return score + bias

        return alibi_mask

    @staticmethod
    @torch.no_grad()
    def build_score_mod(n_heads: int | None, device: torch.device | None) -> None:
        for score_mod_type in FlexAttention.used_score_mod_types:
            if score_mod_type in FlexAttention.score_mods:
                continue

            if score_mod_type == "noop":
                score_mod = FlexAttention._get_noop_score_mod()
            elif score_mod_type in [
                "alibi",
                "alibi_with_nope",
                "linear_alibi",
                "linear_alibi_with_nope",
            ]:
                if n_heads is None or device is None:
                    raise ValueError(
                        "n_heads and device are required for non-noop score_mods."
                    )
                score_mod = FlexAttention._get_alibi_score_mod(
                    heads=n_heads,
                    method=score_mod_type,
                    device=device,
                )
            else:
                raise ValueError(f"Unrecognized score_mod_type {score_mod_type}.")

            FlexAttention.score_mods[score_mod_type] = score_mod

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(
        batch: torch.Tensor, eos_id: int
    ) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        """
        Given an arbitrary mask_mod, divide the input sequence to blocks
        and only allow attention within the same block.

        Args:
            mask_mod: The mask mod to apply to the documents
            fixed_block_size: The number of tokens in each block.
        """

        # Credit to @drisspg.
        def blocked_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Get the block index of the query and key
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            # Only allow attention within the same block
            same_block = q_block == kv_block
            # Apply the original mask mod
            inner_mask = mask_mod(
                b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size
            )

            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
        )

        return blocked_mask_mod

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(
        batch: torch.Tensor, eos_id: int | None = None, max_ctx_len: int | None = None
    ) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, fixed_block_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            if fixed_block_size is not None and fixed_block_size > 0:
                mask_mod = FlexAttention._fixed_block_mask_mod(
                    mask_mod, fixed_block_size
                )

            seq_len = batch.shape[1] if max_ctx_len is None else max_ctx_len
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask


def init_attention_mask(
    batch: torch.Tensor,
    eos_id: int | None = None,
    cp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
    max_ctx_len: int | None = None,
) -> None:
    FlexAttention.init_attention_mask(batch, eos_id, max_ctx_len)


def init_score_mod(n_heads: int, device: torch.device) -> None:
    FlexAttention.build_score_mod(n_heads, device)


class ScaledDotProductAttention(torch.nn.Module):
    backends: ClassVar[list[SDPBackend]] = []
    _bias_buffer: ClassVar[torch.Tensor | None] = None
    _bias_length: ClassVar[int] = 0

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.available_attn_mask_types = [
            "causal",
            "alibi",
            "bidirectional_alibi",
            "linear_alibi",
            "bidirectional_linear_alibi",
            "nope",
            "alibi_with_nope",
            "linear_alibi_with_nope",
        ]
        assert attn_mask_type in self.available_attn_mask_types
        ScaledDotProductAttention._init_backend()

    def get_bias(
        self, kv_len: int, num_heads: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        # update buffer if kv_len exceeds current capacity
        if kv_len > ScaledDotProductAttention._bias_length:
            capacity = kv_len + 136
            ScaledDotProductAttention._bias_buffer = (
                get_attn_mask_for_sdpa(capacity, num_heads, method=self.attn_mask_type)
                .to(device, dtype)
                .unsqueeze(0)
            )
            ScaledDotProductAttention._bias_length = capacity
        return ScaledDotProductAttention._bias_buffer

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        # Add CuDNN on B200 w/ highest priority
        cls.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if has_cuda_capability(10, 0):
            cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    def forward(
        self,
        q: torch.Tensor,  # (B, H, q_len, d)
        k: torch.Tensor,  # (B, H, kv_len, d)
        v: torch.Tensor,
        offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        assert self.backends, "SDPA Backends should not be empty."

        B, H, q_len, _ = q.shape
        kv_len = k.shape[-2]

        is_causal = q_len != 1  # causal false when decoding
        enable_gqa = H != k.shape[1]
        attn_mask = None

        if self.attn_mask_type != "causal":
            bias_full = self.get_bias(kv_len, H, q.device, q.dtype)
            q_start, q_end = kv_len - q_len, kv_len
            attn_mask = bias_full[:, :, q_start:q_end, :kv_len]
            is_causal = False

        with sdpa_kernel(self.backends, set_priority=True):
            # FIXME: should not be needed but getting access errors without it
            q, k, v = map(lambda x: x.contiguous(), (q, k, v))
            attn_mask = attn_mask.contiguous() if attn_mask is not None else None
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=enable_gqa
            )


class AdasplashAttention(torch.nn.Module):
    _slopes_cache: ClassVar[dict[int, torch.Tensor]] = {}
    _varlen_cache: ClassVar[dict[tuple[int, int], torch.Tensor]] = {}

    def __init__(
        self,
        attn_mask_type,
        alpha: float = 1.0,
        block_mask: bool = False,
        n_iter: int = 5,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.alpha = alpha
        self.block_mask = block_mask
        self.niter = n_iter
        self.layer_idx = layer_idx
        self.available_attn_mask_types = [
            "causal",
            "alibi",
            "bidirectional_alibi",
            "linear_alibi",
            "bidirectional_linear_alibi",
            "nope",
            "alibi_with_nope",
            "linear_alibi_with_nope",
        ]
        assert attn_mask_type in self.available_attn_mask_types

    def get_slopes(
        self, num_heads: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if num_heads not in AdasplashAttention._slopes_cache:
            AdasplashAttention._slopes_cache[num_heads] = get_slopes(
                num_heads, self.attn_mask_type
            ).to(device, dtype)
        return AdasplashAttention._slopes_cache[num_heads]

    def get_varlen_cache(
        self, batch_size: int, q_len: int, device: torch.device
    ) -> torch.Tensor:
        if (batch_size, q_len) not in AdasplashAttention._varlen_cache:
            AdasplashAttention._varlen_cache[(batch_size, q_len)] = torch.tensor(
                [q_len] * batch_size, device=device
            )
        return AdasplashAttention._varlen_cache[(batch_size, q_len)]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q: The query tensor of shape (bs, heads, seqlen, dim)
        k: The key tensor of shape (bs, heads, seqlen, dim)
        v: The value tensor of shape (bs, heads, seqlen, dim)
        """
        B, H, q_len, _ = q.shape
        alibi_slopes = None

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if self.attn_mask_type != "causal":
            alibi_slopes = self.get_slopes(H, q.device, q.dtype)

        if offset is not None and q_len > 1:
            # prefill kernel
            attn_output = torch.zeros_like(q)
            attn_output = sparse_attention_prefill(
                q,
                k,
                v,
                attn_output,
                is_causal=True,
                varlen=offset + 1,
                is_left_padding=True,
                alpha=self.alpha,
                niter=6 if self.alpha in [1.5, 2.0] else 10,
                alibi_slopes=alibi_slopes,
            )

        elif offset is not None and q_len == 1:
            # decoding kernel
            attn_output = triton_entmax_attention(
                q,
                k,
                v,
                alpha=self.alpha,
                varlen=offset + 1,
                is_left_padding=True,
                niter=10 if self.alpha in [1.5, 2.0] else 10,
                alibi_slopes=alibi_slopes,
            )

        else:
            # training kernel
            attn_output = adasplash15(
                q,
                k,
                v,
                niter=2,
                alibi_slopes=alibi_slopes,
            )

        return attn_output


def build_attention(
    use_flex_attn: bool,
    attn_mask_type: str,
    fixed_block_size: int | None = None,
    adasplash_alpha: float = 1.0,
    adasplash_block_mask: bool = False,
    adasplash_niter: int = 5,
):
    if use_flex_attn:
        assert adasplash_alpha == 1.0, "FlexAttention only supports alpha=1.0"
        score_mod_type = attn_mask_type if attn_mask_type != "causal" else None
        attn_mask_type = "causal"
        return FlexAttention(
            attn_mask_type=attn_mask_type,
            score_mod_type=score_mod_type,
            fixed_block_size=fixed_block_size,
        )

    elif adasplash_alpha > 1.0:
        return AdasplashAttention(
            attn_mask_type=attn_mask_type,
            alpha=adasplash_alpha,
            block_mask=adasplash_block_mask,
            n_iter=adasplash_niter,
        )

    else:
        if fixed_block_size is not None:
            raise ValueError(
                "TorchTitan with SDPA currently does not support fixed_block_size."
            )
        return ScaledDotProductAttention(attn_mask_type)
