import inspect

import pytest
from transformers import AutoConfig

from caduceus.configuration_caduceus import (CaduceusConfig,
                                             validate_complement_map)
from caduceus.tests.conftest import assert_config_equal, create_test_config


def test_rcps_requires_complement_map():
    with pytest.raises(ValueError, match="A `complement_map` must be provided if rcps=True"):
        CaduceusConfig(rcps=True)


def test_invalid_bidirectional_strategy():
    with pytest.raises(ValueError, match="Unrecognized self.bidirectional_strategy='invalid'; must be one of"):
        CaduceusConfig(bidirectional_strategy="invalid")


def test_invalid_mamba_version():
    with pytest.raises(ValueError, match="Unrecognized self.mamba_version='invalid'; must be one of"):
        CaduceusConfig(mamba_version="invalid")

def test_mamba_version_mismatch():
    with pytest.raises(ValueError, match="Parameter self.mamba_version.* must match the Mamba version implied by self.ssm_cfg"):
        CaduceusConfig(mamba_version="v1", ssm_cfg={"layer": "Mamba2"})


def test_arg_assignment():
    # Check that every parameter in the signature exists as an attribute
    sig = inspect.signature(CaduceusConfig)
    param_names = set(sig.parameters.keys())
    config = CaduceusConfig()
    for param_name in param_names:
        if param_name == "kwargs":
            continue
        assert hasattr(config, param_name), f"Parameter '{param_name}' from constructor not assigned as attribute"


def test_custom_config():
    # Test setting of some non-default values
    config = CaduceusConfig(
        d_model=1024,
        n_layer=32,
        vocab_size=32000,
        bidirectional=False
    )
    assert config.d_model == 1024
    assert config.n_layer == 32
    assert config.vocab_size == 32000
    assert config.bidirectional is False


@pytest.mark.parametrize("vocab_size,multiple,expected", [
    # Multiple of 8
    (100, 8, 104),  # Needs padding to next multiple of 8
    (64, 8, 64),   # Already multiple of 8
    (65, 8, 72),   # Needs padding to next multiple of 8
    
    # Multiple of 2
    (99, 2, 100),  # Needs padding to next multiple of 2
    (100, 2, 100), # Already multiple of 2
    
    # Multiple of 1 (no padding)
    (99, 1, 99),   # No padding needed
    (100, 1, 100), # No padding needed
])
def test_padded_vocab_size(vocab_size, multiple, expected):
    config = CaduceusConfig(vocab_size=vocab_size, pad_vocab_size_multiple=multiple)
    assert config.padded_vocab_size == expected


def test_padded_complement_map():
    # Test when complement_map is smaller than padded_vocab_size
    complement_map = {0: 1, 1: 0, 2: 3, 3: 2}
    config = CaduceusConfig(
        vocab_size=6,
        pad_vocab_size_multiple=2,
        complement_map=complement_map,
        rcps=True
    )
    padded_map = config.padded_complement_map
    assert padded_map[0] == 1
    assert padded_map[1] == 0
    assert padded_map[2] == 3
    assert padded_map[3] == 2
    assert padded_map[4] == 4  # Self-mapping for padding tokens
    assert padded_map[5] == 5  # Self-mapping for padding tokens

    # Test when complement_map equals padded_vocab_size
    complement_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    config = CaduceusConfig(
        vocab_size=6,
        pad_vocab_size_multiple=2,
        complement_map=complement_map,
        rcps=True
    )
    padded_map = config.padded_complement_map
    assert padded_map == complement_map


def test_validate_complement_map():
    # Valid complement map
    valid_map = {0: 1, 1: 0, 2: 3, 3: 2}
    assert validate_complement_map(valid_map) == valid_map

    # Test non-sequential keys
    with pytest.raises(ValueError, match="Complement map keys must be a sorted sequence of integers starting from 0"):
        validate_complement_map({0: 1, 2: 3, 3: 2})  # Missing key 1

    # Test non-zero starting key
    with pytest.raises(ValueError, match="Complement map keys must be a sorted sequence of integers starting from 0"):
        validate_complement_map({1: 2, 2: 1})  # Starts at 1 instead of 0

    # Test duplicate values
    with pytest.raises(ValueError, match="Complement map must not contain duplicate values"):
        validate_complement_map({0: 1, 1: 1, 2: 2})  # Value 1 appears twice

    # Test asymmetric mappings
    with pytest.raises(ValueError, match="Complement map must be symmetric"):
        validate_complement_map({0: 1, 1: 2, 2: 0})  # Forms cycle 0->1->2->0 instead of pairs


def test_complement_map_size_error():
    # Test when complement_map is larger than vocab_size
    complement_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    with pytest.raises(ValueError, match="Vocabulary size must be greater than or equal to the complement map size"):
        CaduceusConfig(
            vocab_size=4,  # Smaller than complement_map size
            complement_map=complement_map,
            rcps=True
        )

def test_autoconfig_roundtrip(tmp_path):
    AutoConfig.register("caduceus", CaduceusConfig)
    original_config = create_test_config()
    config_path = tmp_path / "caduceus-config"
    original_config.save_pretrained(config_path)
    loaded_config = AutoConfig.from_pretrained(config_path)
    assert_config_equal(original_config, loaded_config)

    
