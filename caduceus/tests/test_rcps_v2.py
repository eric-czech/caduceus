from typing import Literal
import inspect
import pytest
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import create_block
from caduceus.tokenization_caduceus import CaduceusTokenizer
from mamba_ssm.modules.mamba_simple import Mamba



def rc(x: torch.Tensor) -> torch.Tensor:
    if x.has_names:
        return torch.flip(x.rename(None), dims=[-2, -1]).rename(*x.names)
    else:
        return torch.flip(x, dims=[-2, -1])
    
class LambdaModule(nn.Module):
    def __init__(self, module, fn):
        super().__init__()
        self.module = module
        self.fn = fn
    def forward(self, *args, **kwargs):
        return self.fn(self.module(*args, **kwargs))

def mamba_layer(config: CaduceusConfig) -> nn.Module:
    return Mamba(d_model=config.d_model)

def bert_layer(config: CaduceusConfig) -> nn.Module:
    layer = BertLayer(BertConfig(
        hidden_size=config.d_model, 
        vocab_size=config.vocab_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        is_decoder=False
    ))
    return LambdaModule(
        module=layer,
        fn=lambda x: x[0]
    )

class CaduceusBlockV2(nn.Module):

    def __init__(self, config: CaduceusConfig, layer: nn.Module):
        super().__init__()
        self.config = config
        self.layer = layer
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.rename("batch", "sequence", "channel")
        # Double the batch with RC sequences
        o = torch.cat([x, rc(x)], dim="batch").rename(None)
        # Process forward and reverse sequences independently
        o = self.layer(self.norm(o))
        # Break complementary sequence pairs back out
        o = o.view(2, *tuple(x.shape)).rename("complement", *x.names)
        # Sum results for each pair (forward + RC)
        o = o.sum(dim="complement")
        assert o.shape == x.shape
        return o


def get_config(n_layer: int):
    tokenizer = CaduceusTokenizer(model_max_length=5)
    return CaduceusConfig(
        d_model=4,
        n_layer=n_layer,
        rcps=True,
        pad_token_id=-100,
        fused_add_norm=False,
        rms_norm=False,
        bidirectional=False,
        vocab_size=tokenizer.vocab_size,
        complement_map=tokenizer.complement_map,
        model_max_length=tokenizer.model_max_length
    )

@pytest.fixture
def device():
    return "cuda"   

def get_caduceus_v1(config: CaduceusConfig, device: str) -> nn.Module:
    blocks = []
    for _ in range(config.n_layer):
        block = create_block(**{
            k: v for k, v in vars(config).items() 
            if k in inspect.signature(create_block).parameters
        }, device=device)
        original_forward = block.forward
        block.forward = lambda *args, **kwargs: original_forward(*args, **kwargs)[0]
        blocks.append(block)
    return nn.Sequential(*blocks)

def get_caduceus_v2(config: CaduceusConfig, layers: Literal["bert", "mamba"] | list[nn.Module], device: str) -> nn.Module:
    blocks = []
    for i in range(config.n_layer):
        if layers == "bert":
            layer = bert_layer(config)
        elif layers == "mamba":
            layer = mamba_layer(config)
        else: 
            layer = layers[i]
        blocks.append(CaduceusBlockV2(config, layer))
    return nn.Sequential(*blocks).to(device)

@pytest.mark.parametrize("n_batch", [4])
def test_v1_v2_equivalence(device, n_batch):
    config = get_config(n_layer=1)
    seq = torch.randn(n_batch, config.model_max_length, config.d_model).to(device)
    caduceus_v1 = get_caduceus_v1(config, device)
    caduceus_v2 = get_caduceus_v2(config, layers=[
        list(caduceus_v1)[i].mixer.submodule.mamba_fwd
        for i in range(config.n_layer)
    ], device=device)

    # Double the sequence on the channels dimension without RC'ing the second half
    # because this is how the inputs would be provided from `RCPSEmbedding`; see:
    # https://github.com/kuleshov-group/caduceus/blob/49e3204dfba5bd36b9c6405bca6b0396e04ab9d2/caduceus/modeling_rcps.py#L65-L67
    expected = caduceus_v1(torch.cat([seq, seq], dim=-1))
    expected = expected[..., :config.d_model] + rc(expected[..., config.d_model:])
    actual = caduceus_v2(seq).rename(None)
    assert torch.allclose(expected, actual)

@pytest.mark.parametrize("n_batch", [1, 4, 8])
@pytest.mark.parametrize("n_layer", [1, 5])
@pytest.mark.parametrize("layer_type", ["bert", "mamba"])
def test_v2_rc_invariance(device, n_batch, n_layer, layer_type):
    config = get_config(n_layer=n_layer)
    seq = torch.randn(n_batch, config.model_max_length, config.d_model).to(device)
    caduceus_v2 = get_caduceus_v2(config, layers=layer_type, device=device)
    assert torch.equal(caduceus_v2(seq), caduceus_v2(rc(seq)))
