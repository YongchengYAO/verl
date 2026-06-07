# Bugfix: PEFT `vocab_size` AttributeError with Qwen2.5-VL

## Error

Training crashed before epoch 0 during the initial validation step:

```
AttributeError: 'Qwen2_5_VLConfig' object has no attribute 'vocab_size'
```

## Call Chain

```
_validate()
  → async_rollout_manager.generate_sequences()
  → wake_up()
  → rollout_mode()
  → collect_lora_params()
  → layered_summon_lora_params()
  → get_peft_model_state_dict()   # crash here
```

## Root Cause

`peft.utils.save_and_load.get_peft_model_state_dict()` defaults to `save_embedding_layers="auto"`, which triggers an automatic check to detect whether the embedding layer's vocabulary size changed during fine-tuning. This check:

1. Reads `vocab_size` from the live model config (returns non-None for Qwen2.5-VL, likely via `text_config` delegation)
2. Loads the original config from HuggingFace Hub via `model.config.__class__.from_pretrained(model_id)`
3. Accesses `.vocab_size` directly on the freshly loaded `Qwen2_5_VLConfig` — which doesn't expose `vocab_size` as a top-level attribute → **AttributeError**

## Fix

**File:** `verl/utils/fsdp_utils.py`

Pass `save_embedding_layers=False` explicitly to skip the vocab_size check at both call sites:

**`layered_summon_lora_params()` (line ~598):**
```python
# Before
sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())

# After
sub_lora_params = get_peft_model_state_dict(
    peft_model, state_dict=submodule.state_dict(), save_embedding_layers=False
)
```

**`collect_lora_params()` (line ~633):**
```python
# Before
lora_params = get_peft_model_state_dict(peft_model)

# After
lora_params = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
```

## Why It's Safe

The training script uses `exclude_modules='.*visual.*'`, so LoRA is applied only to language model layers — no embedding layers are targeted. Skipping embedding layer saving has no effect on correctness.
