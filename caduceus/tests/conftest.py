import inspect
from typing import Literal

import pytest
import torch
from packaging.version import Version

from caduceus.compat.mamba import get_mamba_version
from caduceus.configuration_caduceus import CaduceusConfig


def get_default_mamba_version() -> str:
    # Get the default `mamba_version` from `CaduceusConfig` constructor
    sig = inspect.signature(CaduceusConfig.__init__)
    return sig.parameters['mamba_version'].default

def get_tokenizer_mappings():
    # fmt: off
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }
    # fmt: on
    return str_to_id, complement_map

def get_tokenizer_id(token: str) -> int:
    return get_tokenizer_mappings()[0][token]

def validate_mamba_config(config: CaduceusConfig) -> CaduceusConfig:
    if config.mamba_version == "v2":
        # Check that `d_model * expand / headdim` is a multiple of 8
        ssm_cfg = config.ssm_cfg or {}
        for param in ["headdim", "expand"]:
            if param not in ssm_cfg:
                raise ValueError(
                    f"Parameter {param!r} is required in v2 Mamba configs for testing to "
                    "ensure that `d_model * expand / headdim`; see: "
                    "https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940"
                )
        dim = config.d_model * ssm_cfg["expand"] / ssm_cfg["headdim"]
        if dim % 8 != 0:
            raise ValueError(
                f"For v2 Mamba configs, `d_model * expand / headdim` must be a multiple of 8 (got {dim}); see: "
                "https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940"
            )
    return config


def create_test_config(**kwargs) -> CaduceusConfig:
    # Get tokenizer mappings
    str_to_id, complement_map = get_tokenizer_mappings()

    # Default config values for testing to smaller model size
    defaults = {
        "d_model": 128,
        "n_layer": 2,
        "vocab_size": len(str_to_id),
        "residual_in_fp32": False,
        "pad_vocab_size_multiple": 8,
        "bidirectional": True,
        "bidirectional_strategy": "add",
        "bidirectional_weight_tie": True,
        "rcps": True,
        "complement_map": complement_map,
    }

    if kwargs.get("mamba_version", get_default_mamba_version()) == "v2":
        # Provide defaults for v2 Mamba configs that ensure `d_model * expand / headdim`
        # is a multiple of 8 based off of `d_model`. See:
        # https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940
        if "ssm_cfg" not in kwargs:
            d_model, expand = kwargs.get("d_model", defaults["d_model"]), 2
            headdim = d_model * expand // 8
            kwargs["ssm_cfg"] = {"headdim": headdim, "expand": expand}

    # Override defaults with any provided kwargs
    config_args = {**defaults, **kwargs}
    config = CaduceusConfig(**config_args)
    return validate_mamba_config(config)

def skip_if_mamba_dtype_not_supported(version: Literal["v1", "v2"], device: torch.device, dtype: torch.dtype) -> None:
    # Skip fp32 tests for Mamba v2 on older GPUs due to:
    # https://github.com/triton-lang/triton/issues/4813
    major, _ = torch.cuda.get_device_capability(device)
    if (
        version == "v2" 
        and torch.finfo(dtype).bits >= torch.finfo(torch.float32).bits 
        and major < 8
    ):
        pytest.skip("Mamba v2 fp32+ tests are not supported on this GPU")

def skip_if_mamba_version_not_available(version: Literal["v1", "v2"]) -> None:
    installed_version = get_mamba_version(raise_on_missing=True)
    assert isinstance(installed_version, Version)
    if installed_version.major < 2 and version == "v2":
        pytest.skip("Mamba v2 not installed")

def skip_if_mamba_incompatible(version: Literal["v1", "v2"], device: torch.device, dtype: torch.dtype) -> None:
    skip_if_mamba_version_not_available(version)
    skip_if_mamba_dtype_not_supported(version, device, dtype)

@pytest.fixture
def device() -> torch.device:
    # Assume GPU for all tests now;
    # consider CPU support in the future
    return torch.device("cuda")

@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    # Set seed before any tests are run
    torch.random.manual_seed(0)

def create_random_token_ids(config: CaduceusConfig, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, config.padded_vocab_size, (batch_size, seq_len), device=device)

def assert_config_equal(expected: CaduceusConfig, actual: CaduceusConfig) -> None:
    assert isinstance(expected, CaduceusConfig)
    assert isinstance(actual, CaduceusConfig)
    expected = expected.to_dict()
    actual = actual.to_dict()
    assert expected.keys() == actual.keys()
    for key in expected.keys():
        if key.startswith("_"):
            continue
        assert expected[key] == actual[key], \
            f"Config values for {key=!r} do not match, " \
            f"got {actual[key]=!r} instead of {expected[key]=!r}"