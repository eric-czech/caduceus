from typing import Literal, Tuple

import pytest
import torch
from conftest import (assert_config_equal, create_random_token_ids,
                      create_test_config, get_tokenizer_id,
                      get_tokenizer_mappings, skip_if_mamba_incompatible,
                      skip_if_mamba_version_not_available)
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from caduceus.compat.mamba import MHA, GatedMLP, Mamba2
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import (BiMambaWrapper, Caduceus,
                                        CaduceusForMaskedLM,
                                        CaduceusForSequenceClassification,
                                        RCBatch, RCOperator,
                                        reverse_complement, reverse_sequence,
                                        token_complement_map,
                                        token_reverse_complement)


def get_tokenized_inputs(str_to_id, complement_map, batch_size, seq_len, device):
    fw_input_ids = torch.randint(low=0, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(fw_input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)
    return fw_input_ids, rc_input_ids

def test_token_complement_map():
    # Test valid complement map
    valid_map = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    actual = token_complement_map(valid_map)
    expected = torch.tensor([5, 4, 3, 2, 1, 0])
    assert torch.equal(actual, expected)
    
    # Test invalid complement map (unsorted keys)
    invalid_map = {0: 5, 2: 3, 4: 1}
    with pytest.raises(ValueError, match="Complement map keys must be a sorted sequence of integers"):
        token_complement_map(invalid_map)
    
    # Test invalid complement map (non-zero start)
    invalid_map = {1: 5, 2: 4, 3: 3}
    with pytest.raises(ValueError, match="Complement map keys must be a sorted sequence of integers"):
        token_complement_map(invalid_map)


def test_token_reverse_complement():
    # Setup test data
    complement_map = token_complement_map({0:2, 1:4, 2:0, 3:3, 4:1, 5:5})
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    
    # Test valid input
    actual = token_reverse_complement(complement_map, input_ids)
    # RC(input_ids) = reverse([[2, 4, 0], [3, 1, 5]]) = [[0, 4, 2], [5, 1, 3]]
    expected = torch.tensor([[0, 4, 2], [5, 1, 3]])
    assert torch.equal(actual, expected)
    
    # Test invalid input shape (1D)
    invalid_input = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="Input IDs must be a 2D tensor"):
        token_reverse_complement(complement_map, invalid_input)
    
    # Test invalid input shape (3D)
    invalid_input = torch.tensor([[[0, 1], [2, 3]]])
    with pytest.raises(ValueError, match="Input IDs must be a 2D tensor"):
        token_reverse_complement(complement_map, invalid_input)

    # Test invalid complement map (not one-to-one)
    invalid_map = {0: 1, 1: 2, 2: 0}  # 1 maps to 2, but 2 maps to 0
    with pytest.raises(ValueError, match="Complement map must be symmetric"):
        token_complement_map(invalid_map)
    

def test_tensor_reverse_complement():
    # fmt: off
    # Define embedding channel values for each nucleotide
    A1, A2 = 1, 2  # Channel values for A
    C1, C2 = 3, 4  # Channel values for C
    G1, G2 = 5, 6  # Channel values for G
    T1, T2 = 7, 8  # Channel values for T
    
    # Test tensor reverse complement
    fw_states = torch.tensor([
        [[C1, C2], [G1, G2], [T1, T2]],  # FW seq 1 (C,G,T)
        [[G1, G2], [T1, T2], [A1, A2]],  # FW seq 2 (G,T,A)
    ])
    rc_states = reverse_complement(fw_states)
    
    expected = torch.tensor([
        [[T2, T1], [G2, G1], [C2, C1]],  # Flipped and complemented: T,G,C
        [[A2, A1], [T2, T1], [G2, G1]],  # Flipped and complemented: A,T,G
    ])
    assert torch.equal(rc_states, expected)
    
    # Test with extra dimensions of length 1 at the beginning
    fw_states_5d = fw_states[None, None, None, ...]
    rc_states_5d = reverse_complement(fw_states_5d)
    assert torch.equal(rc_states_5d.squeeze(), expected)
    
    # Test invalid input shape (1D)
    with pytest.raises(ValueError, match="Tensor must have at least 2 dimensions"):
        reverse_complement(torch.tensor([1, 2, 3]))
    # fmt: on


def test_tensor_reverse_sequence():
    # fmt: off
    # Define embedding channel values for each nucleotide
    A1, A2 = 1, 2  # Channel values for A
    C1, C2 = 3, 4  # Channel values for C
    G1, G2 = 5, 6  # Channel values for G
    T1, T2 = 7, 8  # Channel values for T

    # Test tensor reverse sequence
    fw_states = torch.tensor([
        [[C1, C2], [G1, G2], [T1, T2]],  # FW seq 1 (C,G,T)
        [[G1, G2], [T1, T2], [A1, A2]],  # FW seq 2 (G,T,A)
    ])
    rc_states = reverse_sequence(fw_states)
    expected = torch.tensor([
        [[T1, T2], [G1, G2], [C1, C2]],  # Flipped: T,G,C
        [[A1, A2], [T1, T2], [G1, G2]],  # Flipped: A,T,G
    ])
    assert torch.equal(rc_states, expected)

    # Test with extra dimensions of length 1 at the beginning
    fw_states_5d = fw_states[None, None, None, ...]
    rc_states_5d = reverse_sequence(fw_states_5d)
    assert torch.equal(rc_states_5d.squeeze(), expected)

    # Test invalid input shape (1D)
    with pytest.raises(ValueError, match="Tensor must have at least 2 dimensions"):
        reverse_sequence(torch.tensor([1, 2, 3]))
    # fmt: on


def test_rc_batch():
    # fmt: off
    str_to_id, complement_map = get_tokenizer_mappings()
    batch = RCBatch(create_test_config(complement_map=complement_map))
    
    # Define token IDs for clarity
    A, C, G, T = [str_to_id[e] for e in "ACGT"]
    
    # Test forward pass using explicit token IDs
    # Create sequence: [A,C,G] and [T,A,C]
    input_ids = torch.tensor([
        [A, C, G],  # First sequence
        [T, A, C]   # Second sequence
    ])
    
    actual = batch(input_ids)
    expected = torch.tensor([
        [A, C, G],  # Original: A,C,G
        [T, A, C],  # Original: T,A,C
        [C, G, T],  # RC of first: C,G,T
        [G, T, A]   # RC of second: G,T,A
    ])
    assert torch.equal(actual, expected)

    # Test invalid input shapes
    with pytest.raises(ValueError, match="Input IDs must be a 2D tensor"):
        batch(torch.tensor([1, 2, 3]))  # 1D tensor
    
def test_rc_operator():
    # Define embedding channel values for each nucleotide
    A1, A2 = 1, 2  # Channel values for A
    C1, C2 = 3, 4  # Channel values for C
    G1, G2 = 5, 6  # Channel values for G
    T1, T2 = 7, 8  # Channel values for T
    
    # Create a batch of states for two "forward" sequences of length 3
    fw_states = torch.tensor([
        [[A1, A2], [C1, C2], [G1, G2]],  # Original seq 1 (A,C,G)
        [[T1, T2], [A1, A2], [C1, C2]],  # Original seq 2 (T,A,C)
        [[C1, C2], [G1, G2], [T1, T2]],  # RC seq 1 (C,G,T)
        [[G1, G2], [T1, T2], [A1, A2]]   # RC seq 2 (G,T,A)
    ])
    # Create the same batch assuming the reverse complement of 
    # the original sequences were provided instead
    rc_states = torch.stack([
        fw_states[2],  # RC seq 1 (C,G,T)
        fw_states[3],  # RC seq 2 (G,T,A)
        fw_states[0],  # Original seq 1 (A,C,G)
        fw_states[1],  # Original seq 2 (T,A,C)
    ])

    # Test summation
    actual = RCOperator.sum(fw_states)
    expected = torch.tensor([
        [[A1+T1, A2+T2], [C1+G1, C2+G2], [G1+C1, G2+C2]],  # Seq 1 + flipped RC (seq only)
        [[T1+A1, T2+A2], [A1+T1, A2+T2], [C1+G1, C2+G2]],  # Seq 2 + flipped RC (seq only)
    ])
    assert torch.equal(actual, expected)
    
    actual = RCOperator.sum(rc_states)
    assert torch.equal(actual, expected)

    # Test concatenation
    actual = RCOperator.concat(fw_states)
    expected = torch.tensor([
        [[A1,A2,T2,T1], [C1,C2,G2,G1], [G1,G2,C2,C1]], # Seq 1 & flipped RC
        [[T1,T2,A2,A1], [A1,A2,T2,T1], [C1,C2,G2,G1]], # Seq 2 & flipped RC
    ])
    assert torch.equal(actual, expected)

    actual = RCOperator.concat(rc_states)
    actual = reverse_complement(actual)
    assert torch.equal(actual, expected)
    
    with pytest.raises(ValueError, match="Hidden states must be a 3D tensor"):
        RCOperator.sum(torch.tensor([[1, 2], [3, 4]]))  # 2D tensor
    
    with pytest.raises(ValueError, match="Batch size must be divisible by 2"):
        RCOperator.sum(torch.tensor([[[1]], [[2]], [[3]]]))  # Wrong batch size ratio


def test_caduceus_device_placement(device, dtype=torch.float16):
    # Note that dtype for parameters is not validated here
    # because they are not expected to be uniform, e.g.:
    # - Mamba2.A params: https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/mamba2.py#L133
    # - Mamba2.D params: https://github.com/state-spaces/mamba/blob/9182c93c9acb3e4ccac55a18a52c228d870d60bc/mamba_ssm/modules/mamba2.py#L139
    config = create_test_config()
    factory_kwargs = {"device": device, "dtype": dtype}
    model = Caduceus(config, **factory_kwargs)
    
    # Check that all parameters are on the correct device
    for name, param in model.named_parameters():
        expected_device = torch.device(device)
        actual_device = param.device
        assert actual_device.type == expected_device.type, \
            f"Parameter '{name}' is on {actual_device.type}, should be on {expected_device.type}"

    # Check that all buffers are on the correct device
    for name, buffer in model.named_buffers():
        expected_device = torch.device(device)
        actual_device = buffer.device
        assert actual_device.type == expected_device.type, \
            f"Buffer '{name}' is on {actual_device.type}, should be on {expected_device.type}"
        
        
@pytest.fixture(params=[
    {
        "batch_size": 2,
        "seq_len": size,
        "n_layer": layers,
        "d_model": dim,
        "dtype": dtype,
        "bidirectional": bidir,
        "bidirectional_weight_tie": tie,
        "mamba_version": version,
        "rcps": rcps
    }
    for size in [100]
    for layers in [1, 4]
    for dim in [128, 256]
    for dtype in [torch.float32, torch.float16]
    for bidir in [False, True]
    for tie in [False, True]
    for rcps in [False, True]
    for version in ["v1", "v2"]
])
def params(request):
    return request.param

@pytest.fixture
def model_setup(device, params):
    """Common setup for Caduceus model tests."""
    skip_if_mamba_incompatible(params["mamba_version"], device, params["dtype"])
    
    config = create_test_config(
        d_model=params["d_model"],
        n_layer=params["n_layer"],
        residual_in_fp32=True,
        bidirectional=params["bidirectional"],
        bidirectional_weight_tie=params["bidirectional_weight_tie"],
        mamba_version=params["mamba_version"],
        rcps=params["rcps"],
    )
    factory_kwargs = {"device": device, "dtype": params["dtype"]}
    
    str_to_id, complement_map = get_tokenizer_mappings()
    input_ids, rc_input_ids = get_tokenized_inputs(
        str_to_id, complement_map, params["batch_size"], params["seq_len"], device
    )
    
    return config, factory_kwargs, input_ids, rc_input_ids

def assert_rc_equivariance(
    fw_out: torch.Tensor, rc_out: torch.Tensor, 
    expected_shape: Tuple[int, ...], 
    flip_operation: Literal["reverse_sequence", "reverse_complement"]
):
    assert flip_operation in ["reverse_sequence", "reverse_complement"]
    if flip_operation == "reverse_sequence":
        rc_out = reverse_sequence(rc_out)
    elif flip_operation == "reverse_complement":
        rc_out = reverse_complement(rc_out)
    assert tuple(fw_out.size()) == expected_shape
    assert tuple(rc_out.size()) == expected_shape
    assert torch.equal(fw_out, rc_out)

def test_caduceus_backbone(params, model_setup):
    config, factory_kwargs, input_ids, rc_input_ids = model_setup
    model = Caduceus(config, **factory_kwargs)

    fw_out = model(input_ids)
    if not params["rcps"]:
        return
    
    # Check that equivariance holds for forward and RC sequences
    rc_out = model(rc_input_ids)
    fw_out, rc_out = RCOperator.concat(fw_out), RCOperator.concat(rc_out)
    expected_shape = (params["batch_size"], params["seq_len"], params["d_model"]*2)
    assert_rc_equivariance(fw_out, rc_out, expected_shape, "reverse_complement")


def apply_mask(input_ids: torch.Tensor, mask_prob: float = .3) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(input_ids, dtype=torch.float) < mask_prob
    labels = torch.where(~mask, input_ids, -100)
    masked_input_ids = torch.where(mask, torch.tensor(get_tokenizer_id("[MASK]")).to(input_ids.device), input_ids)
    return masked_input_ids, labels

def test_caduceus_language_model(params, model_setup):
    config, factory_kwargs, input_ids, rc_input_ids = model_setup
    model = CaduceusForMaskedLM(config=config, **factory_kwargs)
    
    # Create masked inputs and labels (`mask` sets tokens to predict)
    masked_input_ids, labels = apply_mask(input_ids)
    loss = model(masked_input_ids, labels=labels).loss
    assert not torch.isnan(loss)
    assert float(loss) > 0
    
    if not params["rcps"]:
        return

    # Check that equivariance holds for forward and RC sequences
    fw_out = model(input_ids)
    rc_out = model(rc_input_ids)
    fw_states, rc_states = RCOperator.concat(fw_out.hidden_states), RCOperator.concat(rc_out.hidden_states)
    expected_shape = (params["batch_size"], params["seq_len"], params["d_model"]*2)
    assert_rc_equivariance(fw_states, rc_states, expected_shape, "reverse_complement")

    expected_shape = (params["batch_size"], params["seq_len"], config.padded_vocab_size)
    assert_rc_equivariance(fw_out.logits, rc_out.logits, expected_shape, "reverse_sequence")
    
    assert torch.equal(
        F.softmax(fw_out.logits, dim=-1),
        F.softmax(reverse_sequence(rc_out.logits), dim=-1)
    )

@pytest.mark.parametrize("pooling_strategy", ["mean", "max", "first", "last"])
def test_caduceus_sequence_classifier(params, model_setup, device, pooling_strategy):
    config, factory_kwargs, input_ids, rc_input_ids = model_setup
    model = CaduceusForSequenceClassification(config=config, **factory_kwargs, pooling_strategy=pooling_strategy)
    labels = torch.randint(0, config.num_labels, (params["batch_size"],), device=device)
    fw_out = model(input_ids, labels=labels)
    assert not torch.isnan(fw_out.loss)
    assert float(fw_out.loss) > 0
    
    if not params["rcps"]:
        return
    # These pooling strategies do not support RC equivariance
    if pooling_strategy in ["first", "last"]:
        return
    
    # Check that equivariance holds for forward and RC sequences
    rc_out = model(rc_input_ids, labels=labels)
    fw_states, rc_states = RCOperator.concat(fw_out.hidden_states), RCOperator.concat(rc_out.hidden_states)
    expected_shape = (params["batch_size"], params["seq_len"], params["d_model"]*2)
    assert_rc_equivariance(fw_states, rc_states, expected_shape, "reverse_complement")

    assert torch.allclose(fw_out.logits, rc_out.logits)


@pytest.mark.parametrize("d_intermediate", [16])
@pytest.mark.parametrize("attn_layer_idx", [(1, 3)])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [128])
def test_caduceus_hybrid_architecture(device, d_intermediate, attn_layer_idx, batch_size, seq_len):
    skip_if_mamba_version_not_available("v2")

    config = create_test_config(
        d_model=128,
        n_layer=4,
        d_intermediate=d_intermediate, 
        attn_layer_idx=attn_layer_idx,
        bidirectional=False,
        attn_cfg={"num_heads": 4}
    )
    factory_kwargs = {"device": device, "dtype": torch.float16}
    model = Caduceus(config, **factory_kwargs)
    
    # Validate Mamba or MHA block structure like:
    # Mamba block --> Block(         MHA block --> Block(
    #   (norm): RMSNorm()             (norm): RMSNorm()
    #   (mixer): BiMambaWrapper(      (mixer): MHA(...)
    #     (mamba_fwd): Mamba2(...)
    #   )
    #   (norm2): RMSNorm()            (norm2): RMSNorm()
    #   (mlp): GatedMLP(...)          (mlp): GatedMLP(...)
    # )                             )
    for i, layer in enumerate(model.mixer.layers):
        # Check if block has attention at specified layer indices
        mixer = layer.mixer
        if i in attn_layer_idx:
            assert isinstance(mixer, MHA), f"Layer {i} should be MHA but got {type(mixer)}"
        else:
            assert isinstance(mixer, BiMambaWrapper), f"Layer {i} should be BiMambaWrapper but got {type(mixer)}"
            mixer = mixer.mamba_fwd
            assert isinstance(mixer, Mamba2), f"Layer {i} should be Mamba2 but got {type(mixer)}"
            
        # Check if layer has MLP when d_intermediate > 0
        mlp = layer.mlp
        if d_intermediate > 0:
            assert isinstance(mlp, GatedMLP), f"Layer {i} should have GatedMLP but got {type(mlp)}"
        else:
            assert isinstance(mlp, nn.Identity), f"Layer {i} should have Identity MLP but got {type(mlp)}"

    # Test forward pass
    input_ids = create_random_token_ids(config, batch_size, seq_len, device)
    outputs = model(input_ids)
    expected_shape = (batch_size*2, seq_len, config.d_model)
    assert outputs.shape == expected_shape

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("model_class", [Caduceus, CaduceusForMaskedLM, CaduceusForSequenceClassification])
def test_automodel_roundtrip(tmp_path, device, dtype, model_class):
    AutoConfig.register("caduceus", CaduceusConfig)
    AutoModel.register(CaduceusConfig, model_class)

    # Create model
    config = create_test_config()
    model = model_class(config, device=device, dtype=dtype)
    
    # Create input
    batch_size, seq_len = 1, 8
    input_ids = create_random_token_ids(config, batch_size, seq_len, device=device)
    expected = model(input_ids)

    # Save model
    # Use ``safe_serialization=False`` to avoid error created by bidirectional weight tying:
    # "The weights trying to be saved contained shared tensors"
    # See https://huggingface.co/docs/safetensors/en/torch_shared_tensors.
    model_path = tmp_path / "caduceus-model"
    model.save_pretrained(model_path, safe_serialization=False)

    # Reload and validate
    model = AutoModel.from_pretrained(model_path).to(device, dtype=dtype)
    assert_config_equal(config, model.config)
    actual = model(input_ids)
    if model_class == Caduceus:
        assert torch.equal(actual, expected)
    elif model_class in [CaduceusForMaskedLM, CaduceusForSequenceClassification]:
        actual = model(input_ids)
        assert torch.equal(actual.logits, expected.logits)
        assert torch.equal(actual.hidden_states, expected.hidden_states)
    else:
        raise ValueError(f"Unrecognized model class: {model_class}")
