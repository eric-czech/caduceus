import copy
from typing import Any, Dict, List, Literal, Optional

from transformers import PretrainedConfig


class CaduceusConfig(PretrainedConfig):
    """Configuration class for Caduceus model.

    Args:
        # Caduceus Configuration
        d_model: Hidden dimension of the model (default: 2560)
        n_layer: Number of layers in the model (default: 64)
        rcps: Enable reverse-complement equivariance (default: False)
        complement_map: Token to complement mapping, required if rcps=True (default: None)
        vocab_size: Size of the vocabulary (default: 50277)
        pad_vocab_size_multiple: Pad vocabulary size to be divisible by this value (default: 8)
        bidirectional: Enable bidirectional processing (default: True)
        bidirectional_strategy: How to combine forward/backward sequences ("add" or "ew_multiply") (default: "add")
        bidirectional_weight_tie: Tie weights between input/output projections in each Mamba layer (default: True)
        lm_head_weight_tie: Tie weights for language model head to token embeddings (default: True)

        # Mamba Configuration
        mamba_version: Version of Mamba architecture to use ("v1" or "v2") (default: "v2")
        d_intermediate: Intermediate dimension size for Mamba GatedMLP layers; set to 0 to disable MLPs (default: 0)
            - For Mamba v2.2.4: https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mlp.py#L7-L16
        ssm_cfg: Additional SSM-specific configuration options (default: None)
            - For Mamba v2.2.4: https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2.py#L38-L66
        attn_layer_idx: Layer indices for attention layers (default: None)
        attn_cfg: Additional attention-specific configuration options for Mamba MHA (default: None)
            - For Mamba v2.2.4: https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mha.py#L47-L65
        initializer_cfg: Weight initialization params used by Mamba (default: None)
        residual_in_fp32: Whether to compute residuals in fp32 (default: True)
        fused_add_norm: Use fused add+norm operations (default: True)
        rms_norm: Use RMSNorm instead of LayerNorm (default: True)
        norm_epsilon: Normalization stability term (default: 1e-5)

        # Training Configuration
        gradient_checkpointing_stride: Stride for gradient checkpointing, if enabled (default: 1)

    Example:
        ```python
        from caduceus.tokenization_caduceus import CaduceusTokenizer
        from caduceus.configuration_caduceus import CaduceusConfig

        tokenizer = CaduceusTokenizer()
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
        # Caduceus Configuration
        d_model: int = 2560,
        n_layer: int = 64,
        rcps: bool = False,
        complement_map: Optional[Dict[int|str, int]] = None,
        vocab_size: int = 50277,
        pad_vocab_size_multiple: int = 8,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[Literal["add", "ew_multiply"]] = "add",
        bidirectional_weight_tie: bool = True,
        lm_head_weight_tie: bool = True,
        # Mamba Configuration
        mamba_version: Literal["v1", "v2"] = "v2",
        d_intermediate: int = 0,
        ssm_cfg: Optional[Dict[str, Any]] = None,
        attn_layer_idx: Optional[List[int]] = None,
        attn_cfg: Optional[Dict[str, Any]] = None,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,
        # Training Configuration
        gradient_checkpointing_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.lm_head_weight_tie = lm_head_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
        self.gradient_checkpointing_stride = gradient_checkpointing_stride
        self.mamba_version = mamba_version
        self.d_intermediate = d_intermediate
        self.initializer_cfg = initializer_cfg or {}
        self.ssm_cfg = ssm_cfg or {}
        self.attn_cfg = attn_cfg or {}
        self.attn_layer_idx = attn_layer_idx
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.rms_norm = rms_norm
        self.norm_epsilon = norm_epsilon

        
        # Pre-condition checks
        if "layer" in self.ssm_cfg:
            if self.ssm_cfg["layer"] not in ("Mamba1", "Mamba2"):
                raise ValueError(f"Parameter `layer` must be set to 'Mamba1' or 'Mamba2' for `ssm_cfg`, got {self.ssm_cfg['layer']!r}.")
            implied_mamba_version = "v1" if self.ssm_cfg["layer"] == "Mamba1" else "v2"
            if self.mamba_version != implied_mamba_version:
                raise ValueError(f"Parameter {self.mamba_version=} must match the Mamba version implied by {self.ssm_cfg['layer']=}.")
        if  self.bidirectional and self.bidirectional_strategy not in (vals := ("add", "ew_multiply")):
            raise ValueError(f"Unrecognized {self.bidirectional_strategy=!r}; must be one of {vals}")
        if self.mamba_version not in (vals := ("v1", "v2")):
            raise ValueError(f"Unrecognized {self.mamba_version=!r}; must be one of {vals}")
        if self.attn_layer_idx and self.mamba_version == "v2":
            # Ensure that num_heads is set for MHA because it is a required param:
            # https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mha.py#L50
            # and Mamba does not enforce its existence in a Mixer model here: 
            # https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/models/mixer_seq_simple.py#L63
            if "num_heads" not in self.attn_cfg:
                raise ValueError(
                    "Parameter `num_heads` must be set for `attn_cfg` when attention layers are enabled."
                    f"This is enabled by the existence of the parameter`{self.attn_layer_idx=}`."
                )
        if self.rcps and self.complement_map is None:
            raise ValueError("A `complement_map` must be provided if rcps=True")
        if self.complement_map is not None:
            self.complement_map = parse_complement_map(self.complement_map)
            validate_complement_map(self.complement_map)
            if self.vocab_size < len(self.complement_map):
                raise ValueError(
                    "Vocabulary size must be greater than or equal to the complement map size. "
                    f"Got {self.vocab_size=} and {len(self.complement_map)=}"
                )

        # Computed values
        # See here for how this is used:
        # https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/models/mixer_seq_simple.py#L53-L61
        self.ssm_cfg["layer"] = "Mamba1" if self.mamba_version == "v1" else "Mamba2"


    @property
    def padded_vocab_size(self) -> int:
        vocab_size = self.vocab_size
        if (remainder := vocab_size % self.pad_vocab_size_multiple) != 0:
            vocab_size += self.pad_vocab_size_multiple - remainder
        return vocab_size
    
    @property
    def padded_complement_map(self) -> Optional[dict[int, int]]:
        if self.complement_map is None:
            return None
        return pad_complement_map(self.complement_map, self.padded_vocab_size)

    @property
    def has_mlp(self) -> int:
        return self.d_intermediate > 0

def parse_complement_map(complement_map: dict[int|str, int]) -> dict[int, int]:
    if any(not isinstance(k, int) for k in complement_map):
        complement_map = {int(k): v for k, v in complement_map.items()}
    return complement_map

def validate_complement_map(complement_map: dict[int, int]) -> dict[int, int]:
    # Validate that keys are already sorted integers from 0 to N since
    if list(complement_map.keys()) != list(range(len(complement_map))):
        raise ValueError(f"Complement map keys must be a sorted sequence of integers starting from 0 to {len(complement_map)}. Got keys {list(complement_map.keys())}.")
    # Validate that all keys and values are unique, and form the same set
    if len(set(complement_map.values())) != len(complement_map):
        raise ValueError("Complement map must not contain duplicate values.")
    if list(complement_map.values()) != [complement_map[k] for k in complement_map]:
        raise ValueError("Complement map values must iterate in the same order as keys.")
    # Ensure that mappings are symmetric, i.e. if A -> B, then B -> A
    for k, v in complement_map.items():
        if complement_map[v] != k:
            raise ValueError(f"Complement map must be symmetric. Got {k=}, {v=} with {complement_map[v]=} != {k}.")
    return complement_map

def pad_complement_map(complement_map: dict[int, int], vocab_size: int) -> dict[int, int]:
    complement_map = copy.deepcopy(complement_map)
    for i in range(len(complement_map), vocab_size):
        complement_map[i] = i
    return validate_complement_map(complement_map)