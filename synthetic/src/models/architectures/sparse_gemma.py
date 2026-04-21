import torch, math
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Config,
    Gemma2Model,
    Gemma2ForCausalLM,
    Gemma2DecoderLayer,
    Gemma2Attention,
    Gemma2FlashAttention2,
    Gemma2RotaryEmbedding,
    repeat_kv,
    logger,
    Cache,
    apply_rotary_pos_emb,
    HybridCache
)

import logging
logger = logging.getLogger(__name__)

from entmax import sparsemax, entmax_bisect, entmax15

from transformers.models.gemma2.modeling_gemma2 import (
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10
)

from ...attention.flash_attn import _flash_attention_forward
from ...attention.topk import AttentionNoCache
from ...attention.stickbreaking import sb_attn
from ...attention.stickbreaking.sb_attn_layer import decoding_stickbreaking
from .gptx_rope import GPTNeoXRotaryEmbedding2

from ...kernels.adasplash.adasplash_no_block_mask import sparse_attn
from adasplash import adasplash, adasplash_no_block_mask, triton_entmax


@torch.compile
def polyval(p, x):
    pol_size = p.shape[0]
    p = p.view(-1, *([1]*len(x.shape)))
    pows = torch.arange(pol_size, device=x.device).view(-1, *([1]*len(x.shape)))

    x = x.unsqueeze(0).expand(pol_size, *x.shape)
    x = torch.sum(p * x.pow(pows), dim=0)

    return x


@torch.compile
def adaptive_temperature_softmax(logits):
    """
    "Softmax is not enough". Petar Veličković, Christos Perivolaropoulos, Federico Barbero, Razvan Pascanu
    """
    original_probs = torch.softmax(logits, dim=-1)
    poly_fit = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791]).to(logits.device)
    entropy = torch.sum(-original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdims=True)
    beta = torch.where(
        entropy > 0.5,
        torch.clamp(polyval(poly_fit, entropy), min=1.0),
        1.0
    )

    return torch.softmax(logits * beta, dim=-1)


@torch.compiler.disable
def triton_entmax_no_compile(*args, **kwargs):
   return triton_entmax(*args, **kwargs)


def sparse_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    alpha: float,
    niter: int = 10,
    is_causal: bool = True,
    alibi_slopes=None
):
    # ensure that the input is contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # compute attention
    if attention_mask is None:
        max_seqlen = query.shape[-2]
        varlen = torch.full((query.shape[0],), max_seqlen, dtype=torch.long, device=query.device)
    else:
        varlen = attention_mask.sum(-1).squeeze().long().contiguous()
        if attention_mask.shape[0] == 1:
            varlen = varlen.view(1).contiguous()

    attn_output = sparse_attn(
        query, key, value,
        alpha=alpha,
        varlen=varlen,
        is_causal=is_causal,
        niter=niter,
        alibi_slopes=alibi_slopes
    )

    return attn_output


def get_nape_slopes(num_heads: int, num_alibi_heads: int, slope_type: str = "linear"):
    if slope_type == "linear":
        slopes = 1 / (torch.arange(1, num_alibi_heads + 1, dtype=torch.float32))
    elif slope_type == "exp":
        slopes = 1 / 2 ** (torch.arange(0, num_alibi_heads, dtype=torch.float32))
    else:
        raise NotImplementedError

    return torch.cat((
        slopes,
        torch.zeros(num_heads - num_alibi_heads)
    ))

def make_bias_tensor(slopes: torch.Tensor, max_length: int):
    pos_indices = torch.arange(max_length)
    position_diff = pos_indices[None, :] - pos_indices[:, None]
    position_diff = torch.tril(position_diff)
    # Apply slopes - multiply each head's slope by the position differences
    return (slopes.view(1, -1, 1, 1) * position_diff.view(1, 1, max_length, max_length)).contiguous()


class SparseGemma2Attention(Gemma2Attention):
    def __init__(self, config: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.sliding_window = None
        self.attn_type = getattr(config, "attn_type", "regular")
        self.use_fast_attn = getattr(config, "use_fast_attn", False)
        self.apply_rotary = getattr(config, "apply_rotary", True)
        self.apply_nape = getattr(config, "apply_nape", False)
        self.alibi_slopes = None
        self.entmax_alpha = float(config.entmax_alpha)
        self.qk_scale = 1 / math.sqrt(self.head_dim)

        # Length-Based Adaptive Temperature
        self.attn_scale_type = getattr(config, "attn_scale_type", None)
        self.attn_scale_delta = getattr(config, "attn_scale_delta", 1)
        self.attn_scale_gamma_range = getattr(config, "attn_scale_gamma_range", 2)

        if self.apply_rotary:
            self.rotary_emb = GPTNeoXRotaryEmbedding2(config=config)

        max_seq_len = config.max_position_embeddings

        if self.apply_nape:
            alibi_num_heads = getattr(config, "alibi_num_heads", config.num_attention_heads // 2)
            alibi_slope_type = getattr(config, "alibi_slope_type", "linear")
            self._num_nope_heads = config.num_attention_heads - alibi_num_heads

            self.alibi_slopes = get_nape_slopes(config.num_attention_heads, alibi_num_heads, alibi_slope_type)

            if self.use_fast_attn:
                # Build alibi bias: shape (1, num_heads, 1, max_seq_len)
                positions = torch.arange(-max_seq_len + 1, 1)  # (max_seq_len,)
                alibi_bias = (positions * self.alibi_slopes.unsqueeze(-1))  # (num_heads, max_seq_len)
                alibi_bias = alibi_bias.reshape(1, config.num_attention_heads, 1, -1)
                self.register_buffer("alibi_bias", alibi_bias, persistent=False)
            else:
                self.register_buffer("alibi_bias", make_bias_tensor(self.alibi_slopes, max_seq_len), persistent=False)


        if self.attn_scale_type:
            self.num_scaled_heads = self._num_nope_heads if self.apply_nape else self.num_heads

            if self.num_scaled_heads <= 0:
                raise ValueError(
                    f"num_scaled_heads must be > 0, got {self.num_scaled_heads}. "
                    f"When apply_nape=True, alibi_num_heads must be < num_attention_heads."
                )

            if self.attn_scale_type == "learn":
                self.attn_scale_beta = nn.Parameter(torch.empty(1, self.num_scaled_heads, 1, 1).uniform_(0.1, 1.0))
                self.attn_scale_gamma = nn.Parameter(torch.empty(1, self.num_scaled_heads, 1, 1).normal_(0.0, 0.1))
            elif self.attn_scale_type == "adapt-softplus-tanh":
                attn_scale_proj_bias = getattr(config, "attn_scale_proj_bias", False)
                self.attn_scale_beta_proj = nn.Linear(self.hidden_size, self.num_scaled_heads, bias=attn_scale_proj_bias)
                self.attn_scale_gamma_proj = nn.Linear(self.hidden_size, self.num_scaled_heads, bias=attn_scale_proj_bias)
            elif self.attn_scale_type == "nakanishi":
                self.attn_scale_beta = nn.Parameter(torch.empty(1, self.num_scaled_heads, 1, 1).normal_(1.0, 0.01))
            else:
                raise ValueError(f"Unknown {config.attn_scale_type}")

            # Offset by 2 so positions start at log(2): avoids log(0)=-inf and log(1)=0
            self.register_buffer(
                "log_position",
                torch.log(torch.arange(2, max_seq_len + 2)).reshape(1, 1, -1, 1),
                persistent=False
            )

        if self.attn_type == "regular":
            if self.entmax_alpha == 1.0:
                self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
                if getattr(config, "poly_fit", False):
                    self.attn_func = lambda x: adaptive_temperature_softmax(x)
                else:
                    self.attn_func = lambda x: F.softmax(x, dim=-1)
            elif config.entmax_alpha > 1.0:
                if getattr(config, "use_triton_entmax", False):
                    self.attn_func = lambda x: triton_entmax_no_compile(x, float(config.entmax_alpha))
                else:
                    self.attn_func = lambda x: entmax_bisect(x, float(config.entmax_alpha), dim=-1)
        elif self.attn_type == "topk":
            if not hasattr(config, "topk_size"):
                raise ValueError("attn_type='topk' requires 'topk_size' in config")

            if self.apply_nape:
                self.register_buffer("alibi_bias", make_bias_tensor(self.alibi_slopes, max_seq_len), persistent=False)

            if not hasattr(config, "topk_length_rule"):
                self.topk_length_rule = lambda x: config.topk_size
            elif config.topk_length_rule == "div-2":
                self.topk_length_rule = lambda x: max(config.topk_size, x // 2)

            self.topk_attn = AttentionNoCache(nn.Softmax(-1))
            self.topk_args = {'topk': config.topk_size}
        elif self.attn_type == "stick-break":
            pass
        else:
            raise NotImplementedError


    def _apply_length_scaling(self, query_states, hidden_states, q_len, k_len):
        """Apply length-based adaptive attention temperature to non-ALiBi heads."""
        bsz = query_states.shape[0]
        log_position_slice = self.log_position[:, :, k_len - q_len:k_len, :]
        scaled_queries = query_states[:, -self.num_scaled_heads:].clone()

        if self.attn_scale_type == "nakanishi":
            query_states[:, -self.num_scaled_heads:] = scaled_queries * log_position_slice * self.attn_scale_beta
        else:
            if self.attn_scale_type == "learn":
                attn_scale_beta = self.attn_scale_beta
                attn_scale_gamma = self.attn_scale_gamma
            elif self.attn_scale_type == "adapt-softplus-tanh":
                attn_scale_beta = torch.nn.functional.softplus(self.attn_scale_beta_proj(hidden_states))
                attn_scale_beta = attn_scale_beta.view(bsz, q_len, self.num_scaled_heads, 1).transpose(1, 2)
                attn_scale_gamma = self.attn_scale_gamma_range * torch.tanh(self.attn_scale_gamma_proj(hidden_states))
                attn_scale_gamma = attn_scale_gamma.view(bsz, q_len, self.num_scaled_heads, 1).transpose(1, 2)
            else:
                raise NotImplementedError

            attn_scaler = self.attn_scale_delta + attn_scale_beta * (log_position_slice ** attn_scale_gamma)
            query_states[:, -self.num_scaled_heads:] = scaled_queries * attn_scaler

        return query_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.apply_rotary:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            cos, sin = None, None

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        k_len = key_states.shape[-2]

        if self.attn_scale_type:
            query_states = self._apply_length_scaling(query_states, hidden_states, q_len, k_len)


        if self.attn_type == "regular":
            if self.use_fast_attn and self.entmax_alpha > 1 and query_states.shape == key_states.shape:
                attn_output = sparse_attention_forward(
                    query_states, key_states, value_states, attention_mask,
                    alpha=self.entmax_alpha,
                    alibi_slopes=self.alibi_slopes.to(query_states.device) if self.alibi_slopes is not None else None
                )
            elif self.use_fast_attn and self.entmax_alpha == 1:
                attn_output = self.flash_attention_routine(q_len, query_states, key_states, value_states, attention_mask)
            else:
                # Eager attention: used when use_fast_attn=False, or as a fallback when
                # use_fast_attn=True and entmax_alpha > 1 but q_len != k_len (i.e. during generation)
                query_states = query_states * self.qk_scale
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

                if self.apply_nape:
                    attn_weights = attn_weights + self.alibi_bias[..., -q_len:, -k_len:]

                if attention_mask is not None:
                    # When use_fast_attn=True, the model-level _update_causal_mask returns a raw 2D mask
                    # (flash/triton kernels handle causality internally). But since we fell through to
                    # eager mode (e.g. KV cache shape mismatch), we need to convert it to a 4D causal mask.
                    if self.use_fast_attn:
                        attention_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_value)

                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                # upcast attention to fp32
                attn_weights_fp32 = attn_weights.to(torch.float32)
                attn_weights = self.attn_func(attn_weights_fp32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)

            if not (self.use_fast_attn and self.entmax_alpha == 1):
                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )
                attn_output = attn_output.transpose(1, 2).contiguous()
        elif self.attn_type == "topk":
            attention_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_value)
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]].to(query_states.dtype)
            self.topk_args['topk'] = self.topk_length_rule(k_len)

            if self.apply_nape:
                attention_mask = attention_mask + self.alibi_bias[..., -q_len:, -k_len:]

            attn_output = self.topk_attn(
                query_states * self.qk_scale,
                key_states,
                value_states,
                mask=attention_mask,
                args=self.topk_args
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        elif self.attn_type == "stick-break":
            if q_len == 1:
                attn_output, rem = decoding_stickbreaking(query_states, key_states, value_states)
            else:
                attn_output, rem = sb_attn(query_states, key_states, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = Gemma2Model._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask

    def flash_attention_routine(self, q_len, query_states, key_states, value_states, attention_mask):
        dropout_rate = self.attention_dropout if self.training else 0.0

        if attention_mask is not None:
            seq_len = attention_mask.shape[1]
            key_states = key_states[:, :, :seq_len]
            value_states = value_states[:, :, :seq_len]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.qk_scale,
            is_causal=self.is_causal,
            sliding_window=self.sliding_window,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            softcap=self.config.attn_logit_softcapping if is_flash_attn_greater_or_equal("2.6.0") else None,
            alibi_slopes=self.alibi_slopes.to(query_states.device) if self.alibi_slopes is not None else None,
        )

        return attn_output


class SparseGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: Optional[int] = None):
        config.attn_logit_softcapping = None
        super().__init__(config, layer_idx)
        self.is_sliding = False
        self.self_attn = SparseGemma2Attention(config, layer_idx)


class SparseGemma2Model(Gemma2Model):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [SparseGemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma2 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        use_fast_attn = hasattr(self.config, "use_fast_attn") and self.config.use_fast_attn
        if self.config._attn_implementation == "flash_attention_2" or use_fast_attn:
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


class SparseGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.model = SparseGemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Force DynamicCache during generate() instead of the default HybridCache.
        # HybridCache returns key_states padded to max_cache_len on every step, which
        # breaks the q_len == k_len check in SparseGemma2Attention.forward and routes
        # prefill into the eager fallback with a 1D alibi_bias buffer that is only
        # correct at q_len = 1. DynamicCache grows incrementally, so prefill has
        # q_len == k_len and the adasplash kernel path is taken as intended.
        if self.generation_config is not None:
            self.generation_config.cache_implementation = None

        # Initialize weights and apply final processing
        self.post_init()