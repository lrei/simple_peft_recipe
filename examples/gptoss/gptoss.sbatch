#!/bin/bash
# Slurm template for examples/gptoss: gpt-oss-120b on one node, the model
# split over the node's GPUs (--device_map unsloth_balanced) in one process.
# Replace every <...> placeholder, then: sbatch examples/gptoss/gptoss.sbatch
#
# The 4-bit 120b weights (~58 GiB) do not fit one 40 GB GPU, so data
# parallel (one full copy per GPU) is not possible there; two 40 GB GPUs
# is the minimum, more leave room for longer sequences. On two A100 40GB
# training peaks at ~37 GB per GPU and takes ~41 min with loading and the
# final evaluation (~36 s per step, micro-batch 8 x accumulation 2; README,
# "gpt-oss-120b on two A100 40GB").
# For gpt-oss-20b on one GPU: MODEL=unsloth/gpt-oss-20b-unsloth-bnb-4bit,
# --gres=gpu:1, and drop --device_map and the batch-size flags.

#SBATCH --job-name=speftr-gptoss
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd <path-to-your-clone>
# Model and dataset downloaded beforehand (compute nodes often have no
# internet). Offline, Unsloth needs the unsloth/ model id or a local
# directory, not openai/gpt-oss-120b.
export HF_HOME=<shared-hf-cache>
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

MODEL=${MODEL:-unsloth/gpt-oss-120b-unsloth-bnb-4bit}
OUTPUT_DIR=./models/speftr-gptoss-${SLURM_JOB_ID}

# One process, no torchrun: the layers are spread over all visible GPUs.
uv run python -m examples.gptoss.gptoss \
    --model_name_or_path "$MODEL" \
    --device_map unsloth_balanced \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 2 \
    --output_dir "$OUTPUT_DIR"

uv run --extra gptoss python -m examples.gptoss.gptoss_eval \
    --adapter_dir "$OUTPUT_DIR" --device_map unsloth_balanced --batch_size 4 \
    --max_new_tokens 128 --num_rows 40
