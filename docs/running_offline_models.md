# Running models offline

Download models and datasets once, distribute them internally, and run
with no requests to the Hugging Face Hub.

## Remote code

Some Hub repos ship their own Python (`modeling_*.py`,
`configuration_*.py`, `tokenization_*.py`) and reference it from
`config.json` (`auto_map`). Loading them needs `trust_remote_code=True`,
which downloads and **executes** that code. Models this project uses
(Gemma 3/4, Qwen 3/3.5/3.8) are native to transformers and need none.

For remote-code models:

- Pin a commit (`--revision <sha>`) and review the `.py` files once.
- Install any packages the code imports.
- Download any other repo its `auto_map` references.
- `HF_MODULES_CACHE` (where transformers copies the code) must be
  writable.

## 1. Download once

A plain download fetches everything: weights, tokenizer, chat template
and code.

```bash
hf download google/gemma-4-12B-it --revision <sha> \
    --local-dir /shared/models/gemma-4-12B-it
hf download --repo-type dataset my-org/my-data \
    --local-dir /shared/datasets/my-data
```

Pass the local directory as `model_name_or_path` / `--dataset`.

Alternative: fill a shared cache (`HF_HOME=/shared/hf`) and distribute
that directory; Hub ids then keep working unchanged.

- **Unsloth remaps Google ids.** `PESFT` (Unsloth) loads
  `google/gemma-4-*` ids from the matching `unsloth/*` repo, so offline a
  cached `google/...` id fails with `Unsloth: Failed to load model. Both
  AutoConfig and PeftConfig loading failed`. Download and pass the
  `unsloth/...` id, or pass a local directory.

## 2. Disable network access at runtime

```bash
export HF_HUB_OFFLINE=1            # Hub: cache/local files only
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1       # vLLM usage stats
export DO_NOT_TRACK=1
```

## 3. Verify

Run once without a network; any hidden request fails loudly:

```bash
# No network namespace; needs root (Ubuntu 24.04 blocks `unshare -rn`
# for normal users).
sudo unshare -n -- sudo -u "$USER" -E \
    uv run python -m examples.guard.guard_test --model_path ...
# Or in a container: docker run --network none ...
```

Remote code or libraries (including unsloth) may fetch extra files at
runtime; this test is the only proof they don't.

## Internal mirror

To serve many machines, mirror the Hub with an artifact proxy
(Artifactory, Nexus) and set `HF_ENDPOINT` to it.
