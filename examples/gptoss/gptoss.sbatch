#!/bin/bash
# Slurm template for examples/gptoss: gpt-oss-120b on one node, the model
# split over the node's GPUs (--device_map balanced) in one process.
# Replace every <...> placeholder, then: sbatch examples/gptoss/gptoss.sbatch
#
# The 4-bit 120b weights (~58 GiB) do not fit one 40 GB GPU, so data
# parallel (one full copy per GPU) is not possible there; two 40 GB GPUs
# is the minimum, more leave room for longer sequences. Training on two
# A100 40GB takes ~65 s per step and peaks at ~37 GB per GPU; evaluation
# on two GPUs is not measured yet. The eval batch sizes below are
# estimates from gpt-oss-20b (README, "Memory and time").
# For gpt-oss-20b on one GPU: MODEL=unsloth/gpt-oss-20b-unsloth-bnb-4bit
# and --gres=gpu:1.

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
    --device_map balanced \
    --per_device_eval_batch_size 2 \
    --output_dir "$OUTPUT_DIR"

uv run --extra gptoss python -m examples.gptoss.gptoss_eval \
    --adapter_dir "$OUTPUT_DIR" --device_map balanced --batch_size 8
