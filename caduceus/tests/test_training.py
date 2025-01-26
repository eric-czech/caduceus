import warnings
from collections import defaultdict
from typing import Dict, Literal, Optional

import pytest
import torch
from composer import Trainer as ComposerTrainer
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from conftest import create_test_config
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import CaduceusForMaskedLM

# Reentrant gradient checkpointing is required to force recomputation
# on backward passes, which is necessary for tests that verify checkpointing
# usage based on observing those recomputations
USE_REENTRANT_CHECKPOINTS = True


def create_test_inputs(
    config: CaduceusConfig,
    batch_size: Optional[int] = 2,
    seq_len: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Create random input tensors for testing."""
    shape = (batch_size, seq_len) if batch_size is not None else (seq_len,)
    return {
        'input_ids': torch.randint(0, config.vocab_size, shape, device=device),
        'labels': torch.randint(0, config.vocab_size, shape, device=device)
    }


def create_test_model(
    config: CaduceusConfig,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16
) -> CaduceusForMaskedLM:
    """Create a CaduceusForMaskedLM model with test configuration."""
    model = CaduceusForMaskedLM(config, device=device, dtype=dtype)
    return model


def test_caduceus_mlm(device):
    """Test basic model functionality with default settings used by this test."""
    config = create_test_config()
    model = create_test_model(config, device=device)
    
    # Generate random input
    batch_size, seq_len = 3, 128
    inputs = create_test_inputs(config, batch_size=batch_size, seq_len=seq_len, device=device)
    
    # Run forward pass
    outputs = model(**inputs)
    
    # Check output shapes
    assert outputs.logits.shape == (batch_size, seq_len, model.config.padded_vocab_size)
    
    # Check loss is computed and backpropagates
    assert outputs.loss is not None, "Loss should be computed when labels are provided"
    outputs.loss.backward()
    
    # Check all parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"


def add_forward_count_hooks(model: CaduceusForMaskedLM) -> Dict[int, int]:
    forward_counts: Dict[int, int] = defaultdict(int)

    def count_forwards(module, *args, **kwargs):
        forward_counts[module.layer_idx] += 1

    for layer in model.caduceus.mixer.layers:
        layer.register_forward_hook(count_forwards)

    return forward_counts

def run_direct_pass(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool
) -> None:
    """Run single forward and backward pass explicitly."""
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT_CHECKPOINTS})
    else:
        model.gradient_checkpointing_disable()
        
    outputs = model(**inputs)
    outputs.loss.backward()
    

def run_hf_training(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool,
    output_dir: str
):
    """Run training using HF Trainer and return forward pass counts.
    
    See Also:
        - [Transformers Documentation - Trainer](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.Trainer)
        - [Transformers Source - Trainer implementation](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/trainer.py#L312)
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        disable_tqdm=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT_CHECKPOINTS},
        max_steps=1,
        logging_strategy="no",
        report_to="none",
        # Ensure that serialization is tested as well
        save_strategy="steps",
        save_steps=1,
        # Use safetensors instead of native serialization to avoid errors about 
        # saving shared tensors in different modules (as `BiMambaWrapper` does), e.g.:
        # RuntimeError: The weights trying to be saved contained shared tensors [<tensor_dict>] that are mismatching the transformers base configuration.
        save_safetensors=False,
        # Overwrite must be enabled as pytest tmp_dir is named systematically, e.g.:
        # /tmp/pytest-of-ubuntu/pytest-3/test_gradient_checkpointing_1
        overwrite_output_dir=True, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SimpleDataset(inputs),
    )
    
    trainer.train()

def run_mosaic_training(
    model: CaduceusForMaskedLM,
    inputs: Dict[str, torch.Tensor],
    gradient_checkpointing: bool,
    output_dir: str
) -> Dict[nn.Module, int]:
    """Run training using MosaicML Composer Trainer and return forward pass counts.
    
    See Also:
        - [Composer Documentation - Trainer](https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.Trainer.html)
        - [Composer Source - Trainer implementation](https://docs.mosaicml.com/projects/composer/en/latest/_modules/composer/trainer/trainer.html#Trainer)
    """

    # Wrap the model for Composer
    composer_model = HuggingFaceModel(model)

    optimizer = DecoupledAdamW(model.parameters())

    trainer = ComposerTrainer(
        # Provide optimizer explicitly to avoid warning
        optimizers=optimizer,
        model=composer_model,
        log_to_console=False,
        progress_bar=False,
        train_dataloader=SimpleDataset(inputs),
        precision="amp_fp16",
        max_duration='1ba',
        device_train_microbatch_size=1,
        # Ensure that serialization is tested as well
        save_folder=output_dir,
        save_interval='1ep',
        # Overwrite must be enabled as pytest tmp_dir is named systematically, e.g.:
        # /tmp/pytest-of-ubuntu/pytest-3/test_gradient_checkpointing_1
        save_overwrite=True,
        parallelism_config={
            "fsdp": {
                "activation_checkpointing": gradient_checkpointing, 
                "sharding_strategy": "NO_SHARD"
            }
        }
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict .*")
        trainer.fit()

class SimpleDataset(Dataset):
    """Simple dataset wrapper for a single input (or single batch of inputs)."""
    
    def __init__(self, inputs: Dict[str, torch.Tensor]):
        self.inputs = inputs
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        return self.inputs

@pytest.mark.parametrize("gradient_checkpointing_stride", [1, 2])
@pytest.mark.parametrize("gradient_checkpointing", [True, False])
@pytest.mark.parametrize("mode", ["direct", "huggingface", "mosaic"])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gradient_checkpointing_recomputation(
    gradient_checkpointing_stride: int,
    gradient_checkpointing: bool,
    mode: Literal["direct", "huggingface", "mosaic"],
    tmp_path,
    device,
    dtype
):
    """Validate gradient checkpointing by counting recomputations of checkpointed layers."""
    config = create_test_config(
        gradient_checkpointing_stride=gradient_checkpointing_stride,
        n_layer=4,
    )
    
    # Generate random input
    # HF Trainer requires that inputs for this model have no batch dim and be on CPU first;
    # errors like this are raised for inputs created directly off-CPU:
    # RuntimeError: cannot pin 'torch.cuda.LongTensor' only dense CPU tensors can be pinned
    inputs = create_test_inputs(
        config, 
        batch_size=None if mode == "huggingface" else 1, 
        device=None if mode == "huggingface" else device
    )

    model = create_test_model(config=config, device=device, dtype=dtype)
    if mode == "mosaic":
        # Parameters in Mamba modules are not created with uniform dtypes by design (some are pinned to fp32),
        # so even when a Caduceus model is created with a lower precision dtype, mixed dtypes will
        # be present and must be coerced to a uniform type for Mosaic to avoid:
        # `ValueError: Must flatten tensors with uniform dtype but got torch.float16 and torch.float32`
        model = model.to(dtype=dtype)
    forward_counts = add_forward_count_hooks(model)
    
    # Run training
    run_fn = (run_hf_training if mode == "huggingface" 
             else run_mosaic_training if mode == "mosaic"
             else run_direct_pass)
    kwargs = {"output_dir": str(tmp_path)} if mode in ["huggingface", "mosaic"] else {}
    run_fn(model, inputs, gradient_checkpointing, **kwargs)
    
    # Verify counts
    for layer in model.caduceus.mixer.layers:
        layer_idx = layer.layer_idx
        if gradient_checkpointing and layer_idx % gradient_checkpointing_stride == 0:
            # Checkpointed layers should be computed twice
            assert forward_counts[layer_idx] == 2
        else:
            # Non-checkpointed layers should be computed once
            assert forward_counts[layer_idx] == 1
