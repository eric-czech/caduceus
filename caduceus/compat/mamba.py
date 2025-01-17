from importlib.metadata import PackageNotFoundError, version as pkg_version
from packaging.version import parse, Version

def get_mamba_version(raise_on_missing: bool=False) -> Version | None:
    try:
        
        return parse(pkg_version('mamba-ssm'))
    except PackageNotFoundError as e:
        if raise_on_missing:
            raise PackageNotFoundError(
                "Failed to determine Mamba version. "
                "Verify that `mamba-ssm` is installed."
            ) from e
        return None


def get_mamba_modules():
    """Get mamba normalization modules based on available version.
    
    Returns:
        tuple: (Block, RMSNorm, layer_norm_fn, rms_norm_fn)
    """
    version = get_mamba_version()
    if version is None:
        return (None,)*5
    if version.major < 2:
        from mamba_ssm.modules.mamba_simple import Block
        from mamba_ssm.ops.triton.layernorm import (  # v1 structure
            RMSNorm, 
            layer_norm_fn, 
            rms_norm_fn
        )
        return Block, None, RMSNorm, layer_norm_fn, rms_norm_fn
    else:
        from mamba_ssm.modules.block import Block  
        from mamba_ssm.modules.mlp import GatedMLP
        from mamba_ssm.ops.triton.layer_norm import (  # v2 structure
            RMSNorm, 
            layer_norm_fn, 
            rms_norm_fn
        )
        return Block, GatedMLP, RMSNorm, layer_norm_fn, rms_norm_fn

Block, GatedMLP, RMSNorm, layer_norm_fn, rms_norm_fn = get_mamba_modules()