"""Caduceus config for Hugging Face.

"""

from typing import Literal, Optional, Union, TypeVar, Dict, Any

from transformers import PretrainedConfig


T = TypeVar('T', bound=PretrainedConfig)
ConfigType = Union[Dict[str, Any], T]


class CaduceusMambaConfig(PretrainedConfig):
    """Configuration for Mamba SSM layers.

    For example `ssm_cfg` options, see:
      - v1: [mamba_ssm/modules/mamba_simple.py#L31-L50](https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/mamba_simple.py#L31-L50)
      - v2: [mamba_ssm/modules/mamba2.py#L37-L66](https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/mamba2.py#L37-L66)
    """
    def __init__(
        self,
        version: Literal["v1", "v2"] = "v1",
        ssm_cfg: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.version = version
        self.ssm_cfg = ssm_cfg


class CaduceusLayerConfig(PretrainedConfig):
    def __init__(
        self,
        mode: Literal["mamba-only"] = "mamba-only",
        mamba_cfg: Optional[ConfigType[CaduceusMambaConfig]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.mamba_cfg = CaduceusMambaConfig(**mamba_cfg) if isinstance(mamba_cfg, dict) else mamba_cfg or CaduceusMambaConfig()


class CaduceusNormConfig(PretrainedConfig):
    def __init__(
        self,
        fused_add_norm: bool = True,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fused_add_norm = fused_add_norm
        self.rms_norm = rms_norm
        self.norm_epsilon = norm_epsilon


class CaduceusInitializerConfig(PretrainedConfig):
    def __init__(
        self,
        rescale_prenorm_residual: bool = True,
        initializer_range: float = 0.02,
        n_residuals_per_layer: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.initializer_range = initializer_range
        self.n_residuals_per_layer = n_residuals_per_layer


class CaduceusConfig(PretrainedConfig):
    """Configuration class for Caduceus model.

    Args:
        # Model Architecture
        d_model: Hidden dimension of the model (default: 2560)
        n_layer: Number of layers in the model (default: 64)
        vocab_size: Size of the vocabulary (default: 50277)
        pad_vocab_size_multiple: Pad vocabulary size to be divisible by this value (default: 8)
        residual_in_fp32: Whether to compute residuals in fp32 (default: True)

        # Bidirectional Processing
        bidirectional: Enable bidirectional processing (default: True)
        bidirectional_strategy: How to combine forward/backward sequences ("add" or "ew_multiply") (default: "add")
        bidirectional_weight_tie: Tie weights between input/output projections (default: True)
        
        # Reverse-Complement Processing
        rcps: Enable reverse-complement equivariance (default: False)
        complement_map: Token to complement mapping, required if rcps=True (default: None)
        
        # Component Configurations
        initializer_cfg: Weight initialization settings
            rescale_prenorm_residual: Rescale residuals by 1/√N (default: True)
            initializer_range: Weight initialization std dev (default: 0.02)
            n_residuals_per_layer: Residual connections per layer (default: 1)
        
        layer_cfg: Layer settings
            mode: Layer architecture type (default: "mamba-only")
            mamba_cfg: Mamba-specific settings (default: CaduceusMambaConfig()):
                version: Mamba version (default: "v2")
                ssm_cfg: Optional SSM configuration (default: None)

        norm_cfg: Normalization settings
            fused_add_norm: Use fused add+norm operations (default: True)
            rms_norm: Use RMSNorm instead of LayerNorm (default: True)
            norm_epsilon: Normalization stability term (default: 1e-5)

    Example:
        ```python
        config = CaduceusConfig(
            d_model=2560,
            n_layer=64,
            vocab_size=tokenizer.vocab_size,
            complement_map=tokenizer.complement_map,
            rcps=True
        )
        ```
    """
    model_type = "caduceus"

    def __init__(
            self,
            d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            residual_in_fp32: bool = True,
            pad_vocab_size_multiple: int = 8,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[Literal["add", "ew_multiply"]] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[Dict[str, Any]] = None,
            initializer_cfg: Optional[ConfigType[CaduceusInitializerConfig]] = None,
            layer_cfg: Optional[ConfigType[CaduceusLayerConfig]] = None,
            norm_cfg: Optional[ConfigType[CaduceusNormConfig]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map

        # Handle config objects that might be dictionaries for nested configs; see:
        # - Nesting config implementation: [transformers/configuration_utils.py#L891](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/configuration_utils.py#L891)
        # - Nested config example (CLIP): [clip/configuration_clip.py#L100](https://github.com/huggingface/transformers/blob/bef7dded22a2800908d034540d49837e9ef6fd39/src/transformers/models/clip/configuration_clip.py#L100)
        self.initializer_cfg = CaduceusInitializerConfig(**initializer_cfg) if isinstance(initializer_cfg, dict) else initializer_cfg or CaduceusInitializerConfig()
        self.layer_cfg = CaduceusLayerConfig(**layer_cfg) if isinstance(layer_cfg, dict) else layer_cfg or CaduceusLayerConfig()
        self.norm_cfg = CaduceusNormConfig(**norm_cfg) if isinstance(norm_cfg, dict) else norm_cfg or CaduceusNormConfig()

