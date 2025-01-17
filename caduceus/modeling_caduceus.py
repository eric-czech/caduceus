"""Caduceus model for Hugging Face.

"""

import math
from functools import partial
from typing import Any, Optional, Protocol, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

from .compat.mamba import Block as MambaBlock, GatedMLP as MambaMLP, RMSNorm, get_mamba_version, layer_norm_fn, rms_norm_fn
from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock


def create_block(
    config: CaduceusConfig,
    layer_idx: int,
    device=None,
    dtype=None,
):
    """Create Caduceus block.

    Args:
        config: CaduceusConfig object containing model configuration
        layer_idx: Index of the current layer
        device: Device to create the block on
        dtype: Data type for the block parameters
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        BiMambaWrapper,
        config=config,
        layer_idx=layer_idx,
        **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not config.norm_cfg.rms_norm else RMSNorm,
        eps=config.norm_cfg.norm_epsilon,
        **factory_kwargs
    )

    # For reference on these arguments, see:
    # https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/block.py
    block_cls = RCPSMambaBlock if config.rcps else MambaBlock
    block_args = dict(
        dim=config.d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=config.norm_cfg.fused_add_norm,
        residual_in_fp32=config.residual_in_fp32,
    )
    if get_mamba_version(raise_on_missing=True).major >= 2:
        mlp_cfg = config.layer_cfg.mamba_cfg.mlp_cfg
        if mlp_cfg is None:
            mlp_cls = nn.Identity
        else:
            # For more details on Mamba's GatedMLP, see:
            # https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/mlp.py#L6-L17
            mlp_cls = partial(
                MambaMLP, out_features=config.d_model, **mlp_cfg, **factory_kwargs
            )
        block_args["mlp_cls"] = mlp_cls
    block = block_cls(**block_args)
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
        self,
        # This is typically referred to as `d_model`, but we use `dim`
        # for compatibility with Mamba `mixer_cls` values that are invoked
        # with a single positional argument called `dim`. See:
        # https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/block.py#L30
        # `BiMambaWrapper` is used as a `mixer_cls` argument in `create_block`.
        dim: int, 
        *,
        config: CaduceusConfig,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        """Initialize BiMambaWrapper.
        
        Args:
            config: Configuration object containing model parameters
            layer_idx: Index of the current layer
            device: Device to create the module on
            dtype: Data type for the module parameters
        """
        super().__init__()
        self.bidirectional = config.bidirectional
        self.bidirectional_strategy = config.bidirectional_strategy
        block_cls = Mamba2 if config.layer_cfg.mamba_cfg.version == "v2" else Mamba
        factory_kwargs = {"device": device, "dtype": dtype}
        mamba_kwargs = config.layer_cfg.mamba_cfg.ssm_cfg or {}
        if dim != config.d_model:
            raise AssertionError(
                f"Expected `dim` to be equal to `config.d_model`; "
                f"got {dim=} and {config.d_model=}."
            )
        self.mamba_fwd = block_cls(
            d_model=config.d_model,
            layer_idx=layer_idx,
            **mamba_kwargs,
            **factory_kwargs
        )
        if config.bidirectional:
            self.mamba_rev = block_cls(
                d_model=config.d_model,
                layer_idx=layer_idx,
                **mamba_kwargs,
                **factory_kwargs
            )
            if config.bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"{self.config.bidirectional_strategy=!r}")
        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size, config.d_model, config.complement_map, **factory_kwargs
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)


class HFGCProtocol(Protocol):
    """Protocol for modules that support gradient checkpointing with Hugging Face Transformers."""

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        """Enable gradient checkpointing.
        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        
        See Also:
            - [Transformers Documentation - enable gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable)
            - [Transformers Source - enable implementation](https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/modeling_utils.py#L2521)
        """
        ...

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing.
        
        See Also:
            - [Transformers Documentation - disable gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_disable)
            - [Transformers Source - disable implementation](https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/modeling_utils.py#L2585)
        """
        ...


class MCGCProtocol(Protocol):
    """Protocol for modules that support gradient checkpointing with MosaicML Composer."""

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        """Determine if module should be checkpointed.
        
        See Also:
            - [Composer Documentation - FSDP auto wrap policy](https://github.com/mosaicml/composer/blob/7fa03545cc2025f256d914abc111a068d239d632/docs/source/notes/distributed_training.rst#composers-fsdp-auto-wrap-policy)
            - [MosaicML Examples - GPT implementation](https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L173-L175)
        """
        ...


class CaduceusMixerModel(nn.Module, HFGCProtocol, MCGCProtocol):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.norm_cfg.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.norm_cfg.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.gradient_checkpointing = False
        self.gradient_checkpointing_stride = config.gradient_checkpointing_stride
        if not (1 <= config.gradient_checkpointing_stride <= config.n_layer):
            raise ValueError(
                f"`gradient_checkpointing_stride` must be between 1 and {config.n_layer}; "
                f"got {config.gradient_checkpointing_stride=}."
            )
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    config=config,
                    layer_idx=i,
                    **factory_kwargs
                )
                for i in range(config.n_layer)
            ]
        )

        norm_f = (nn.LayerNorm if not config.norm_cfg.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_cfg.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.norm_cfg.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)

    def _gradient_checkpointing_indexes(self) -> list[int]:
        return [
            i for i in range(len(self.layers)) 
            if i % self.gradient_checkpointing_stride == 0
        ]

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        for index in self._gradient_checkpointing_indexes():
            if self.layers[index] is module:
                return True
        return False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None


    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        checkpoint_indexes = set(self._gradient_checkpointing_indexes())
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_fn = layer
            if self.gradient_checkpointing and index in checkpoint_indexes:
                layer_fn = partial(
                    torch.utils.checkpoint.checkpoint, 
                    layer, **self.gradient_checkpointing_kwargs
                )
            hidden_states, residual = layer_fn(
                # Only positional args can be used for `torch.utils.checkpoint.checkpoint`
                hidden_states, residual, None # inference_params=None
            )

        if not self.fused_add_norm:
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(self, module, **kwargs):
        """Initialize weights.
        
        Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
        """
        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg
        rescale_prenorm_residual = initialized_cfg.rescale_prenorm_residual
        initializer_range = initialized_cfg.initializer_range
        n_residuals_per_layer = initialized_cfg.n_residuals_per_layer

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class Caduceus(CaduceusPreTrainedModel, HFGCProtocol):
    """Caduceus model that can be instantiated using HF patterns."""
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        if config.rcps:
            assert config.complement_map is not None, "Complement map must be provided for RCPS."

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        if config.complement_map is not None and config.vocab_size > len(config.complement_map):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = CaduceusMixerModel(config, **factory_kwargs, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class CaduceusForMaskedLM(CaduceusPreTrainedModel, HFGCProtocol):
    """HF-compatible Caduceus model for masked language modeling."""

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,  # Use caduceus config as it might have been updated
                vocab_size=self.config.vocab_size,  # Use caduceus config as it might have been updated
                true_dim=config.d_model,
                dtype=dtype
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,  # Use caduceus config as it might have been updated
                bias=False,
                **factory_kwargs
            )

        # Initialize weights and apply final processing
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.caduceus.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.caduceus.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Overrides output embeddings."""
        if self.config.rcps:
            raise NotImplementedError("Setting output embeddings for RCPS LM is not supported.")
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie weights, accounting for RCPS."""
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.caduceus

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
        self.caduceus = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights)
            else:
                loss = cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel, HFGCProtocol):
    def __init__(
            self,
            config: CaduceusConfig,
            pooling_strategy: str = "mean",
            conjoin_train: bool = False,
            conjoin_eval: bool = False,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
        self.post_init()
        self.init_scorer()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.caduceus.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.caduceus.gradient_checkpointing_disable()

    def init_scorer(self):
        initializer_range = self.config.initializer_cfg.initializer_range
        self.score.weight.data.normal_(std=initializer_range)

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get hidden representations from the backbone
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., :self.config.d_model],
                    torch.flip(transformer_outputs[0][..., self.config.d_model:], dims=[1, 2])
                 ],
                dim=-1
            )
        elif self.conjoin_train or (self.conjoin_eval and not self.training):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert input_ids.ndim == 3, "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            transformer_outputs = self.caduceus(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.caduceus(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack([transformer_outputs[0], transformer_outputs_rc[0]], dim=-1)
        else:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if hidden_states.ndim == 4:  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
