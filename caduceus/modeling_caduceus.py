from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from mamba_ssm.distributed.gradient_checkpoints import (HFGCProtocol,
                                                        MCGCProtocol)
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.models.mixer_seq_simple import \
    _init_weights as mamba_init_weights
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)

from .configuration_caduceus import CaduceusConfig, validate_complement_map


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



def token_complement_map(complement_map: dict[int, int], dtype=torch.long, device=None) -> torch.Tensor:
    complement_map = validate_complement_map(complement_map)
    return torch.tensor(list(complement_map.values()), dtype=dtype, device=device)

def token_reverse_complement(complement_map: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"Input IDs must be a 2D tensor with shape (batch_size, sequence_length). Got {input_ids.shape=}.")
    return torch.gather(
        complement_map.expand(input_ids.shape[0], -1),
        dim=1,
        index=torch.flip(input_ids, dims=[-1])
    )

def reverse_complement(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions. Got {tensor.shape=}.")
    # Assume shape (..., sequence_length, channels)
    return torch.flip(tensor, dims=[-2, -1])

def reverse_sequence(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions. Got {tensor.shape=}.")
    # Assume shape (..., sequence_length, channels)
    return torch.flip(tensor, dims=[-2])


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality.
    
    See here for more context: https://github.com/state-spaces/mamba/issues/99
    """

    def __init__(
        self,
        dim: int, 
        *,
        config: CaduceusConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        **ssm_cfg: Any,
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
        block_cls = Mamba2 if config.mamba_version == "v2" else Mamba
        factory_kwargs = {"device": device, "dtype": dtype}
        if dim != config.d_model:
            raise AssertionError(
                f"Expected `dim` to be equal to `config.d_model`; "
                f"got {dim=} and {config.d_model=}."
            )
        # Remove `layer` from ssm_cfg since for compatibility with:
        # https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/models/mixer_seq_simple.py#L53-L61
        block_kwargs = {k: v for k, v in ssm_cfg.items() if k != "layer"}
        self.mamba_fwd = block_cls(
            d_model=config.d_model,
            layer_idx=layer_idx,
            **block_kwargs,
            **factory_kwargs
        )
        if config.bidirectional:
            self.mamba_rev = block_cls(
                d_model=config.d_model,
                layer_idx=layer_idx,
                **block_kwargs,
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
        """Forward pass for bidirectional Mamba blocks.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, num_channels)
            inference_params: Optional Mamba-specific inference parameters; see:
                https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/utils/generation.py#L18-L34
        Returns:
            Tensor of shape (batch_size, sequence_length, num_channels)
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                reverse_sequence(hidden_states),  # Flip along the sequence length dimension
                inference_params=inference_params
            )
            out_rev = reverse_sequence(out_rev) # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"{self.bidirectional_strategy=!r}")
        return out



class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
        self, 
        module,
        **kwargs,
    ):
        n_residuals_per_layer = 2 if self.config.has_mlp else 1
        config = self.config.initializer_cfg
        params = {}
        if "initializer_range" in config:
            params["initializer_range"] = config["initializer_range"]
        if "rescale_prenorm_residual" in config:
            params["rescale_prenorm_residual"] = config["rescale_prenorm_residual"]
        mamba_init_weights(
            module,
            n_layer=self.config.n_layer,
            n_residuals_per_layer=n_residuals_per_layer,
            **params,
        )

def assert_shape(tensor: torch.Tensor, **expected_shape: Dict[str, int]) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != tuple(expected_shape.values()):
        raise AssertionError(
            f"Tensor with shape {actual_shape} does not match expected shape "
            f"({', '.join([f'{k}={v}' for k, v in expected_shape.items()])})."
        )

class RCOperator:
    """Operations on DNA sequence features that maintain reverse complement (RC) equivariance.
    
    This class manages splitting sequences into original and reverse complement representations
    as well as merging those representations back together. Many binary functions, in principle,
    can be applied to perform this merge operation while still maintaining RC equivariance.
    This class provides semantics for a few of those.

    For example, assume we have a sequence $x_{fw}$ and its reverse complement $x_{rc}$. 
    RC equivariance requires that any function $f$ applied to $x_{fw}$ will produce the
    same output as $f$ applied to $x_{rc}$, if and only if $f(x_{rc})$ is further transformed
    in a manner that is consistent with the reverse complement operator. I.e. outputs must
    be transformed using a similar operation to the input operation (the definition of equivariance).
    A "similar operation" means one of two things in this context:

    - Flipping a reverse complement sequence along the sequence length dimension alone
    - Flipping a reverse complement sequence along the sequence length and channel dimensions

    The first is sufficient if the binary function used to merge seqeuence representations
    (by this class) is symmetric / commutative, e.g. addition. The second is necessary if that
    binary function is not symmetric, e.g. concatenation.

    For more information, see: 
    - Shiff et al. 2024 https://arxiv.org/abs/2403.03234
    - https://github.com/kuleshov-group/caduceus/issues/66
    """

    @classmethod
    def batch(cls, input_ids: torch.LongTensor, complement_map: torch.Tensor) -> torch.Tensor:
        """Batch tokens as original and reverse complement sequences.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            complement_map: Complement map of shape (vocab_size,)
        Returns: 
            Tensor of shape (2*batch_size, sequence_length)
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Input IDs must be a 2D tensor with shape (batch_size, sequence_length). Got {input_ids.shape=}.")
        rc_input_ids = token_reverse_complement(complement_map, input_ids)
        result = torch.cat([input_ids, rc_input_ids], dim=0)
        assert_shape(
            result, 
            batch_size=input_ids.shape[0]*2,
            sequence_length=input_ids.shape[1],
        )
        return result
    
    @classmethod
    def split(cls, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split hidden states into original and reverse complement states.
        
        Args:
            hidden_states: Tensor of shape (2*batch_size, sequence_length, num_channels)
        Returns:
            Tuple of tensors of shape (batch_size, sequence_length, num_channels)
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"Hidden states must be a 3D tensor. Got {hidden_states.shape=}.")
        if hidden_states.shape[0] % 2 != 0:
            raise ValueError(f"Batch size must be divisible by 2. Got {hidden_states.shape[0]=}.")
        batch_size = hidden_states.shape[0] // 2
        # Split into original and reverse complement states
        fw_states, rc_states = hidden_states[:batch_size], hidden_states[batch_size:]
        return fw_states, rc_states

    @classmethod
    def sum(cls, hidden_states: torch.Tensor, flip: bool = True) -> torch.Tensor:
        """Sum hidden states for original and reverse complement sequences.
        
        Args:
            hidden_states: Tensor of shape (2*batch_size, sequence_length, num_channels)
            flip: Whether to reverse sequences along the length dimension
        Returns:
            Tensor of shape (batch_size, sequence_length, num_channels)
        """
        return cls.apply(
            hidden_states,
            merge_operation=lambda x, y: x + y,
            flip_operation=reverse_sequence if flip else None,
            shape_change=(1/2, 1, 1), # batch halves
        )
    
    @classmethod
    def concat(cls, hidden_states: torch.Tensor, flip: bool = True) -> torch.Tensor:
        """Concatenate hidden states for original and reverse complement sequences.
        
        Args:
            hidden_states: Tensor of shape (2*batch_size, sequence_length, num_channels)
            flip: Whether to reverse sequences and channels
        Returns:
            Tensor of shape (batch_size, sequence_length, 2*num_channels)
        """
        return cls.apply(
            hidden_states,
            merge_operation=lambda x, y: torch.cat([x, y], dim=-1),
            flip_operation=reverse_complement if flip else None,
            shape_change=(1/2, 1, 2), # batch halves, channels double
        )

    @classmethod
    def flip(cls, hidden_states: torch.Tensor, complement: bool) -> torch.Tensor:
        """Flip hidden states for original and reverse complement sequences.
        
        Args:
            hidden_states: Tensor of shape (2*batch_size, sequence_length, num_channels)
            complement: Whether to flip along the complement map
        Returns:
            Tensor of shape (2*batch_size, sequence_length, num_channels)
        """
        return cls.apply(
            hidden_states,
            merge_operation=lambda x, y: torch.cat([x, y], dim=0),
            flip_operation=reverse_complement if complement else reverse_sequence,
        )
    
    @classmethod
    def apply(
        cls, 
        hidden_states: torch.Tensor, 
        merge_operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        flip_operation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shape_change: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        fw_states, rc_states = cls.split(hidden_states)
        if flip_operation is not None:
            rc_states = flip_operation(rc_states)
        result = merge_operation(fw_states, rc_states)
        expected_shape = tuple(hidden_states.shape)
        if shape_change is not None:
            expected_shape = tuple(int(e * c) for e, c in zip(expected_shape, shape_change))
        assert_shape(
            result, 
            batch_size=expected_shape[0],
            sequence_length=expected_shape[1],
            num_channels=expected_shape[2],
        )
        return result
    
class RCBatch(nn.Module):

    def __init__(self, config: CaduceusConfig, device=None):
        super().__init__()
        self.config = config
        # Get token index mapping as a (1, vocab_size) tensor
        mapping = token_complement_map(config.padded_complement_map, device=device).unsqueeze(0)
        assert mapping.shape == (1, config.padded_vocab_size)
        self.register_buffer("complement_map", mapping)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass for batching tokens as original and reverse complement sequences.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
        Returns:
            Tensor of shape (2*batch_size, sequence_length)
        """
        return RCOperator.batch(input_ids, self.complement_map)

    

def raise_on_inputs_embeds(inputs_embeds: Optional[torch.FloatTensor]) -> None:
    if inputs_embeds is not None:
        # Note: this would need to be implemented in the MambaMixer model first
        raise NotImplementedError("Using `inputs_embeds` is not supported; use `input_ids` instead.")


class GradientCheckpointingMixin(HFGCProtocol, MCGCProtocol):
    """Mixin for gradient checkpointing support."""

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return self.gradient_checkpointing_module.activation_checkpointing_fn(module)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        self.gradient_checkpointing_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_module.gradient_checkpointing_disable()

    @property
    def gradient_checkpointing_module(self) -> nn.Module:
        """Override this property in child classes to specify which module to checkpoint."""
        raise NotImplementedError
    

class Caduceus(CaduceusPreTrainedModel, GradientCheckpointingMixin):
    """Caduceus model that can be instantiated using HF patterns."""

    def __init__(self, config: CaduceusConfig, *, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch = RCBatch(config, device=device) if config.rcps else None
        mixer_type = partial(BiMambaWrapper, config=config)
        self.mixer = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=config.padded_vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=config.initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            gradient_checkpointing_stride=config.gradient_checkpointing_stride,
            mixer_type=mixer_type,
            **factory_kwargs
        )
        self.config = config
        
    @property
    def gradient_checkpointing_module(self) -> nn.Module:
        return self.mixer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Caduceus.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            inputs_embeds: Optional input embeddings of shape (batch_size, sequence_length, d_model)
        Returns:
            A tensor of shape (2*batch_size, sequence_length, d_model) if reverse complement processing
            is enabled or a tensor of shape (batch_size, sequence_length, d_model) otherwise.
        """
        raise_on_inputs_embeds(inputs_embeds)

        if self.config.rcps:
            input_ids = self.batch(input_ids)
        hidden_states = self.mixer(input_ids)
        assert_shape(
            hidden_states, 
            batch_size=input_ids.shape[0], 
            sequence_length=input_ids.shape[1], 
            num_channels=self.config.d_model,
        )
        return hidden_states


class CaduceusForMaskedLM(CaduceusPreTrainedModel, GradientCheckpointingMixin):
    """HF-compatible Caduceus model for masked language modeling."""

    def __init__(self, config: CaduceusConfig, *, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.lm_head = nn.Linear(
            config.d_model,
            config.padded_vocab_size,
            bias=False,
            **factory_kwargs
        )
        self.post_init()

    @property
    def gradient_checkpointing_module(self) -> nn.Module:
        return self.caduceus
    
    def tie_weights(self):
        if self.config.lm_head_weight_tie:
            self.lm_head.weight = self.caduceus.mixer.embedding.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
    ) -> MaskedLMOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            inputs_embeds: Optional input embeddings of shape (batch_size, sequence_length, d_model)
            labels: Optional long tensor of shape (batch_size,)
                - Values should be in `[0, ..., config.padded_vocab_size - 1]` with the exception 
                  of sentinel values to ignore in loss calculation (typically `-100`).
            loss_weights: Optional float tensor of shape (..., config.padded_vocab_size)
        """
        raise_on_inputs_embeds(inputs_embeds)

        hidden_states = self.caduceus(input_ids=input_ids, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states)
        if self.config.rcps:
            # Sum logits across forward and reverse sequences
            logits = RCOperator.sum(logits)

        assert_shape(
            logits,
            batch_size=input_ids.shape[0],
            sequence_length=input_ids.shape[1],
            vocab_size=self.config.padded_vocab_size,
        )

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights)
            else:
                loss = cross_entropy(logits, labels)
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel, GradientCheckpointingMixin):
    def __init__(
        self,
        config: CaduceusConfig,
        *,
        pooling_strategy: str = "mean",
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.config = config
        self.pooling_strategy = pooling_strategy
        self.num_labels = config.num_labels
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False, **factory_kwargs)
        self.post_init()
        self.init_scorer()

    @property
    def gradient_checkpointing_module(self) -> nn.Module:
        return self.caduceus

    def init_scorer(self, initializer_range=0.02):
        initializer_range = self.config.initializer_cfg.get("initializer_range", initializer_range) \
            if self.config.initializer_cfg is not None else initializer_range
        self.score.weight.data.normal_(std=initializer_range)

    def pool_hidden_states(self, hidden_states: torch.Tensor, sequence_length_dim: int = 1) -> torch.Tensor:
        """Pools hidden states along sequence length dimension."""
        match self.pooling_strategy:
            case "mean":
                return hidden_states.mean(dim=sequence_length_dim, keepdim=True)
            case "max":
                return hidden_states.max(dim=sequence_length_dim, keepdim=True)[0]  # [0] to get values, not indices
            case "last":
                return hidden_states.select(dim=sequence_length_dim, index=-1).unsqueeze(sequence_length_dim)
            case "first":
                return hidden_states.select(dim=sequence_length_dim, index=0).unsqueeze(sequence_length_dim)
            case _:
                raise NotImplementedError(f"Pooling strategy `{self.pooling_strategy}` not implemented.")

    def compute_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        loss = None
        if self.config.problem_type == "regression":
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                loss = F.mse_loss(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        """Forward pass for sequence classification.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            inputs_embeds: Optional input embeddings of shape (batch_size, sequence_length, d_model)
            labels: Optional labels of shape (batch_size,)
                - For classification, a long tensor with values in `[0, config.num_labels - 1]`.
                - For regression, a float tensor
        """
        raise_on_inputs_embeds(inputs_embeds)

        hidden_states = self.caduceus(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if self.config.rcps:
            # Flip along the sequence dimension before pooling to ensure that
            # non-commutative pooling operations like `first` and `last` are correct.
            flipped_states = RCOperator.flip(hidden_states, complement=False)
            logits = self.score(self.pool_hidden_states(flipped_states))
            # Average logits for forward and reverse sequences without flipping
            # in the sequence dimension since that already happened above.
            logits = RCOperator.sum(logits, flip=False) / 2
        else:
            logits = self.score(self.pool_hidden_states(hidden_states))

        assert_shape(
            logits,
            batch_size=input_ids.shape[0],
            sequence_length=1,
            num_labels=self.num_labels,
        )
        logits = logits.squeeze(dim=1)
        
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
    
