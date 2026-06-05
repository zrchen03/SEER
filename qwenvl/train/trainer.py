import os
import re
from typing import Dict, List, Optional, Sequence, Union, Tuple

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler, Dataset, IterableDataset, RandomSampler, SequentialSampler
import transformers
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,
    Qwen2VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)
from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_utils import seed_worker
from torch.nn import CrossEntropyLoss

from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

GEN_EMB_ID = 151669

# Memory queue configuration
MEMORY_QUEUE_SIZE = 512


class MemoryQueue:
    """
    FIFO feature queue for storing historical batch features as additional negatives.
    Supports distributed training: features are gathered across GPUs before enqueuing.
    """

    def __init__(self, feature_dim: int, queue_size: int = 256, device: str = 'cuda'):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.device = device

        # Queue is lazily initialized on first enqueue
        self.queue = None
        self.ptr = 0          # current write position
        self.is_full = False  # whether the queue is full

    def _init_queue(self, device, dtype=torch.float32):
        """Lazy-initialize the queue."""
        if self.queue is None:
            self.queue = torch.zeros(self.queue_size, self.feature_dim, device=device, dtype=dtype)
            self.device = device
            self.dtype = dtype

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor):
        """
        Enqueue features (FIFO).

        Args:
            features: [batch_size, feature_dim] tensor (should be detached)
        """
        # Lazy-initialize using the same dtype as input features
        self._init_queue(features.device, features.dtype)

        # Gather features across GPUs so all queues stay in sync
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features.contiguous())
            features = torch.cat(gathered_features, dim=0)

        batch_size = features.shape[0]

        # If batch is larger than queue, keep only the latest queue_size entries
        if batch_size > self.queue_size:
            features = features[-self.queue_size:]
            batch_size = self.queue_size

        # Write into the queue
        if self.ptr + batch_size <= self.queue_size:
            self.queue[self.ptr:self.ptr + batch_size] = features
        else:
            # Wrap around to the beginning (FIFO)
            overflow = (self.ptr + batch_size) - self.queue_size
            self.queue[self.ptr:] = features[:batch_size - overflow]
            self.queue[:overflow] = features[batch_size - overflow:]
            self.is_full = True

        self.ptr = (self.ptr + batch_size) % self.queue_size

        # Mark as full once the queue wraps
        if self.ptr == 0 and batch_size > 0:
            self.is_full = True

    def get_queue(self) -> Optional[torch.Tensor]:
        """
        Return valid features from the queue.

        Returns:
            Tensor of shape [valid_size, feature_dim], or None if empty.
        """
        if self.queue is None:
            return None

        if self.is_full:
            return self.queue.clone()
        elif self.ptr > 0:
            # Queue not yet full; return only the filled portion
            return self.queue[:self.ptr].clone()
        else:
            return None

    def __len__(self):
        """Return the number of valid features in the queue."""
        if self.queue is None:
            return 0
        return self.queue_size if self.is_full else self.ptr


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Flash attention forward for Qwen3VL (from qwen3_trainer.py)."""
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.unsqueeze(0)
    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Qwen2VL attention forward (from qwen3_trainer.py)."""
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3VL attention forward (from qwen3_trainer.py)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_qwen3(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    """
    Replace attention forward functions for packed sequence training (data_flatten mode).
    Following qwen3_trainer.py approach.
    """
    import transformers

    # Qwen2VL
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen2vl_attn_forward
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = return_mask
    # Qwen2.5VL
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2vl_attn_forward
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = return_mask
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = return_mask
    # Qwen3VL
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3vl_attn_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask
    # Qwen3VL MoE
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = qwen3vl_attn_forward
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = return_mask


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Qwen3VLModel has language_model, Qwen2/2.5VLModel has embed_tokens directly
    llm = self.language_model if hasattr(self, 'language_model') else self

    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in llm.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(llm.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None:
        train_dataset = self.train_dataset
    if train_dataset is None or not has_length(train_dataset):
        return None

    if self.args.data_group:
        print("Using SequentialSampler for training dataset.")
        return SequentialSampler(train_dataset)

    # Build the sampler.
    if self.args.group_by_length:
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            lengths = (
                train_dataset[self.args.length_column_name]
                if self.args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = (
            self.processing_class.model_input_names[0] if self.processing_class is not None else None
        )
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    else:
        return RandomSampler(train_dataset)


def gather_features(
        query_features,
        target_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_query_features = hvd.allgather(query_features)
            all_target_features = hvd.allgather(target_features)
        else:
            with torch.no_grad():
                all_query_features = hvd.allgather(query_features)
                all_target_features = hvd.allgather(target_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_query_features = list(all_query_features.chunk(world_size, dim=0))
                gathered_target_features = list(all_target_features.chunk(world_size, dim=0))
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
                all_query_features = torch.cat(gathered_query_features, dim=0)
                all_target_features = torch.cat(gathered_target_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_query_features = torch.cat(torch.distributed.nn.all_gather(query_features), dim=0)
            all_target_features = torch.cat(torch.distributed.nn.all_gather(target_features), dim=0)
        else:
            gathered_query_features = [torch.zeros_like(query_features) for _ in range(world_size)]
            gathered_target_features = [torch.zeros_like(target_features) for _ in range(world_size)]
            dist.all_gather(gathered_query_features, query_features)
            dist.all_gather(gathered_target_features, target_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
            all_query_features = torch.cat(gathered_query_features, dim=0)
            all_target_features = torch.cat(gathered_target_features, dim=0)

    return all_query_features, all_target_features


class ClipLossWithMemory(nn.Module):
    """
    Contrastive loss with memory queue. Same interface as a standard CLIP loss but augments
    both query and target sides with historical features from the memory queue as additional negatives.
    """

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, query_features, target_features, query_memory_features, target_memory_features, logit_scale):
        """
        Compute logits for both directions, augmenting each side with its memory queue features.

        Args:
            query_features: current batch query features
            target_features: current batch target (positive) features
            query_memory_features: historical query features (used for target→query direction)
            target_memory_features: historical target features (used for query→target direction)
            logit_scale: temperature scaling factor
        """
        if self.world_size > 1:
            all_query_features, all_target_features = gather_features(
                query_features, target_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            # Concatenate memory features (symmetric dual-queue)
            if target_memory_features is not None and target_memory_features.shape[0] > 0:
                target_memory_features = target_memory_features.to(all_target_features.dtype)
                all_target_features_with_memory = torch.cat([all_target_features, target_memory_features], dim=0)
            else:
                all_target_features_with_memory = all_target_features

            if query_memory_features is not None and query_memory_features.shape[0] > 0:
                query_memory_features = query_memory_features.to(all_query_features.dtype)
                all_query_features_with_memory = torch.cat([all_query_features, query_memory_features], dim=0)
            else:
                all_query_features_with_memory = all_query_features

            if self.local_loss:
                logits_per_query = logit_scale * query_features @ all_target_features_with_memory.T
                logits_per_target = logit_scale * target_features @ all_query_features_with_memory.T
            else:
                logits_per_query = logit_scale * all_query_features @ all_target_features_with_memory.T
                logits_per_target = logit_scale * all_target_features @ all_query_features_with_memory.T
        else:
            # Single-GPU case
            if target_memory_features is not None and target_memory_features.shape[0] > 0:
                target_memory_features = target_memory_features.to(target_features.dtype)
                target_features_with_memory = torch.cat([target_features, target_memory_features], dim=0)
            else:
                target_features_with_memory = target_features

            if query_memory_features is not None and query_memory_features.shape[0] > 0:
                query_memory_features = query_memory_features.to(query_features.dtype)
                query_features_with_memory = torch.cat([query_features, query_memory_features], dim=0)
            else:
                query_features_with_memory = query_features

            logits_per_query = logit_scale * query_features @ target_features_with_memory.T
            logits_per_target = logit_scale * target_features @ query_features_with_memory.T

        return logits_per_query, logits_per_target

    def forward(self, query_features, target_features, query_memory_features=None, target_memory_features=None, logit_scale=50.0, output_dict=False):
        """
        Compute symmetric contrastive loss, augmented with memory queue negatives.

        Args:
            query_features: [N, dim] current batch query features
            target_features: [N, dim] current batch positive (target) features
            query_memory_features: [M, dim] historical query features (for target→query direction)
            target_memory_features: [M, dim] historical target features (for query→target direction)
            logit_scale: temperature scaling factor
            output_dict: if True, return a dict; otherwise return a scalar
        """
        device = query_features.device
        logits_per_query, logits_per_target = self.get_logits(
            query_features, target_features, query_memory_features, target_memory_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_query.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_query, labels) +
            F.cross_entropy(logits_per_target, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def get_embedding_reps(last_hidden_state, input_ids, embedding_token_ids):
    """
    Extract embeddings for one or more special token IDs.
    Reuses original single-token logic per ID, then stacks and mean pools.
    Returns: (pooled [B, dim], individual [B, K, dim])
    """
    if isinstance(embedding_token_ids, int):
        embedding_token_ids = [embedding_token_ids]

    batch_size = last_hidden_state.shape[0]
    device = last_hidden_state.device

    all_reps = []
    for tid in embedding_token_ids:
        embedding_idx = input_ids == tid
        embedding_idx = torch.where(embedding_idx, torch.arange(input_ids.shape[1], device=device), -1)
        embedding_idx = torch.max(embedding_idx, dim=1).values
        reps = last_hidden_state[torch.arange(batch_size, device=device), embedding_idx]
        all_reps.append(reps)

    individual = torch.stack(all_reps, dim=1)  # [B, K, dim]
    pooled = individual.mean(dim=1)             # [B, dim]
    return pooled, individual


def forward(
    self,
    qry: Optional[dict] = None,
    pos: Optional[dict] = None,
):
    qry['output_hidden_states'] = True
    pos['output_hidden_states'] = True

    qry_output = self.single_forward(**qry)
    pos_output = self.single_forward(**pos)

    # Extract generative embedding representations (multi-token mean pooling)
    gen_emb_ids = self.gen_emb_ids if hasattr(self, 'gen_emb_ids') else [GEN_EMB_ID]

    gen_qry_reps, _ = get_embedding_reps(qry_output.hidden_states[-1], qry['input_ids'], embedding_token_ids=gen_emb_ids)
    gen_pos_reps, _ = get_embedding_reps(pos_output.hidden_states[-1], pos['input_ids'], embedding_token_ids=gen_emb_ids)

    gen_qry_reps = torch.nn.functional.normalize(gen_qry_reps, p=2, dim=-1)
    gen_pos_reps = torch.nn.functional.normalize(gen_pos_reps, p=2, dim=-1)

    rank = torch.distributed.get_rank(group=None)
    world_size = torch.distributed.get_world_size(group=None)

    # Gen loss with memory queue (symmetric dual-queue)
    gen_qry_memory_features = None
    gen_pos_memory_features = None
    if hasattr(self, 'gen_qry_memory_queue') and self.gen_qry_memory_queue is not None:
        gen_qry_memory_features = self.gen_qry_memory_queue.get_queue()
    if hasattr(self, 'gen_pos_memory_queue') and self.gen_pos_memory_queue is not None:
        gen_pos_memory_features = self.gen_pos_memory_queue.get_queue()

    gen_loss_fct = ClipLossWithMemory(
        local_loss=True,
        gather_with_grad=True,
        cache_labels=False,
        rank=rank,
        world_size=world_size,
        use_horovod=False
    )
    gen_contrastive_loss = gen_loss_fct(
        gen_qry_reps,
        gen_pos_reps,
        query_memory_features=gen_qry_memory_features,
        target_memory_features=gen_pos_memory_features,
        logit_scale=50
    )

    # Update memory queues
    if hasattr(self, 'gen_qry_memory_queue') and self.gen_qry_memory_queue is not None:
        self.gen_qry_memory_queue.enqueue(gen_qry_reps.detach())
    if hasattr(self, 'gen_pos_memory_queue') and self.gen_pos_memory_queue is not None:
        self.gen_pos_memory_queue.enqueue(gen_pos_reps.detach())

    contrastive_loss = gen_contrastive_loss
    loss = contrastive_loss + qry_output.loss + pos_output.loss

    return {
        "loss": loss,
        "contrastive_loss": contrastive_loss,
        "qry_loss": qry_output.loss,
        "pos_loss": pos_output.loss,
        "gen_contrastive_loss": gen_contrastive_loss,
    }


def single_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, _= self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def forward_qwen2vl(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if inputs_embeds is None:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.visual.dtype)
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
            video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.model.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.model.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.model.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                delta = delta.to(position_ids.device)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
    )


def single_forward_qwen3vl(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
    """
    Qwen3VL single forward pass. Uses self.language_model instead of self.model.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.visual.dtype)
            image_embeds, _ = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
            video_embeds, _ = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.model.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
    )


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer.
    """
    INITIAL_SKIP = []

    # Auto-Healing: memoized step-skip logic.
    # On initialization reads the training log to detect the last crash step and skips it.
    if not hasattr(self, '_cached_skip_steps'):
        self._cached_skip_steps = set(INITIAL_SKIP)

        log_path = os.getenv('CURRENT_TRAIN_LOG_FILE')
        is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

        if is_rank0 and log_path:
            history_file = log_path + ".skipped"

            # Load existing skip history
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        for line in f:
                            step = int(line.strip())
                            self._cached_skip_steps.add(step)
                    print(f"[Auto-Healing] Loaded skip history: {sorted(list(self._cached_skip_steps))}")
                except Exception as e:
                    print(f"[Auto-Healing] Failed to read history file: {e}")

            # Parse log to find the latest crash step
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'rb') as f:
                        f.seek(0, 2)
                        read_size = 100 * 1024 * 1024
                        f.seek(max(0, f.tell() - read_size), 0)
                        content = f.read().decode('utf-8', errors='ignore')

                    found_step = None
                    lines = content.splitlines()

                    for line in reversed(lines):
                        if "Loading checkpoint" in line: continue

                        matches = re.findall(r'(\d+)/(\d+)', line)
                        valid_match = False
                        for curr_str, total_str in matches:
                            curr = int(curr_str)
                            total = int(total_str)

                            if total < 100: continue
                            if curr <= 1: continue
                            if "it/s" not in line and "%" not in line and "s/it" not in line: continue

                            found_step = curr
                            valid_match = True
                            print(f"\n[Auto-Healing] Detected crash at step: {curr}")
                            break
                        if valid_match: break

                    if found_step is not None:
                        new_skips = {found_step, found_step + 1}

                        if not new_skips.issubset(self._cached_skip_steps):
                            self._cached_skip_steps.update(new_skips)
                            try:
                                with open(history_file, 'w') as f:
                                    for s in sorted(list(self._cached_skip_steps)):
                                        f.write(f"{s}\n")
                                print(f"[Auto-Healing] Updated skip history: {history_file}")
                            except Exception as e:
                                print(f"[Auto-Healing] Failed to write history file: {e}")

                        print(f"[Auto-Healing] Active skip list: {sorted(list(self._cached_skip_steps))}")
                    else:
                        print(f"[Auto-Healing] No new crash step found in log.")

                except Exception as e:
                    print(f"[Auto-Healing] Log parsing failed: {e}")

    device = next(model.parameters()).device
    current_step = self.state.global_step

    should_skip = torch.tensor([0], device=device, dtype=torch.int32)

    if current_step in self._cached_skip_steps:
        should_skip[0] = 1

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(should_skip, op=torch.distributed.ReduceOp.MAX)

    if should_skip[0] > 0:
        if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
            print(f"[SKIP] Step {current_step} skipped (Auto-Healing).")

        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if return_outputs:
            return dummy_loss, {"loss": dummy_loss}
        return dummy_loss

    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if (
        self.args.average_tokens_across_devices
        and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        and num_items_in_batch is not None
    ):
        loss *= self.accelerator.num_processes

    self.log(
        {
            "loss": loss.item(),
            "contrastive_loss": outputs["contrastive_loss"].item(),
            "qry_loss": outputs["qry_loss"].item(),
            "pos_loss": outputs["pos_loss"].item(),
            "gen_contrastive_loss": outputs["gen_contrastive_loss"].item(),
        }
    )
    return (loss, outputs) if return_outputs else loss


# Apply monkey patches
Trainer.create_optimizer = create_optimizer
Trainer.compute_loss = compute_loss
Trainer._get_train_sampler = _get_train_sampler
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VLForConditionalGeneration.forward = forward
Qwen2_5_VLForConditionalGeneration.single_forward = single_forward

Qwen2VLForConditionalGeneration.forward = forward
Qwen2VLForConditionalGeneration.single_forward = forward_qwen2vl

# Qwen3VL monkey patches
Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLForConditionalGeneration.forward = forward
Qwen3VLForConditionalGeneration.single_forward = single_forward_qwen3vl
# Qwen3VL MoE monkey patches
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeForConditionalGeneration.forward = forward
Qwen3VLMoeForConditionalGeneration.single_forward = single_forward_qwen3vl
