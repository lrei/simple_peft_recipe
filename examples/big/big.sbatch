#!/bin/bash
# Slurm template for examples/big on one node with several GPUs.
# Replace every <...> placeholder, then: sbatch examples/big/big.sbatch
#
# MODE=ddp   (default) one process per GPU, each with a full model copy;
#            needs the 4-bit model (~20 GB) to fit on every GPU.
# MODE=split one process, the model's layers spread over all GPUs
#            (--device_map balanced); for models too large for one GPU.
#
# Both modes are implemented but not yet validated on multi-GPU hardware.

#SBATCH --job-name=speftr-big
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd <path-to-your-clone>
# Models and datasets downloaded beforehand; compute nodes often have no
# internet. Offline, pass the unsloth/ model id or a local directory.
export HF_HOME=<shared-hf-cache>
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

MODE=${MODE:-ddp}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
OUTPUT_DIR=./models/speftr-big-${SLURM_JOB_ID}

if [ "$MODE" = "ddp" ]; then
    # Effective batch = 1 per GPU x accumulation x GPUs; keep it at 16.
    uv run torchrun --standalone --nproc_per_node "$NUM_GPUS" \
        -m examples.big.big \
        --gradient_accumulation_steps $((16 / NUM_GPUS)) \
        --output_dir "$OUTPUT_DIR"
else
    uv run python -m examples.big.big --device_map balanced \
        --output_dir "$OUTPUT_DIR"
fi
