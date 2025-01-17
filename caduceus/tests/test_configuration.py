import pytest
from caduceus.configuration_caduceus import CaduceusConfig


def test_custom_config():
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


def test_rcps_requires_complement_map():
    with pytest.raises(ValueError, match="A `complement_map` must be provided if rcps=True"):
        CaduceusConfig(rcps=True)


def test_invalid_bidirectional_strategy():
    with pytest.raises(NotImplementedError, match="Unrecognized self.bidirectional_strategy='invalid'"):
        CaduceusConfig(bidirectional_strategy="invalid")

def test_invalid_mamba_version():
    with pytest.raises(NotImplementedError, match="Unrecognized self.layer_cfg.mamba_cfg.version='invalid'"):
        CaduceusConfig(
            layer_cfg={
                "mamba_cfg": {
                    "version": "invalid"
                }
            }
        )

def test_nested_configs():
    config = CaduceusConfig(
        layer_cfg={
            "mode": "mamba-only",
            "mamba_cfg": {
                "version": "v2",
                "ssm_cfg": {"example": "value"},
                "mlp_cfg": {"hidden_features": 1024}
            }
        }
    )
    assert config.layer_cfg.mode == "mamba-only"
    assert config.layer_cfg.mamba_cfg.version == "v2"
    assert config.layer_cfg.mamba_cfg.ssm_cfg == {"example": "value"}
    assert config.layer_cfg.mamba_cfg.mlp_cfg == {"hidden_features": 1024}


def test_nested_configs_always_present():
    config = CaduceusConfig()
    # Test initializer_cfg defaults
    assert config.initializer_cfg.rescale_prenorm_residual is True
    assert config.initializer_cfg.initializer_range == 0.02
    
    # Test layer_cfg defaults
    assert config.layer_cfg.mode == "mamba-only"
    assert config.layer_cfg.mamba_cfg.version == "v2"
    assert config.layer_cfg.mamba_cfg.ssm_cfg is None
    
    # Test norm_cfg defaults
    assert config.norm_cfg.fused_add_norm is True
    assert config.norm_cfg.rms_norm is True
    assert config.norm_cfg.norm_epsilon == 1e-5


