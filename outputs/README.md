# Outputs

Training writes checkpoints to `outputs/<run_name>/` (or `runs/<run_name>/`); these are **gitignored**
(the final checkpoint is ~370 MB).

## Get the trained checkpoint

```bash
huggingface-cli download ziyad-m97/nano4m-audio checkpoint-final.safetensors \
    --local-dir outputs/animal_full_5mod_v5
```

or in Python:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="ziyad-m97/nano4m-audio", filename="checkpoint-final.safetensors")
```

## The reported checkpoint

- Run: `animal_full_5mod_v5_fresh`, config `cfgs/nano4M/animal_full_5mod_v5.yaml`.
- 18,311 steps, batch 64, ~600 M tokens, ~1h10 on 1× H100, fp32.
- File: `checkpoint-final.safetensors` (~95.84 M parameters).

> To upload: `huggingface-cli upload ziyad-m97/nano4m-audio
> runs/animal_full_5mod_v5_fresh/checkpoint-final.safetensors checkpoint-final.safetensors`.
> Add a model card linking back to this repo, the config, and the eval metrics in `eval_results/`.
