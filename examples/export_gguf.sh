#!/usr/bin/env bash
# Export a merged 16-bit model to GGUF for llama.cpp and Ollama.
#
# Usage (from the repo root, in the project environment plus gguf):
#   uv run --with gguf examples/export_gguf.sh MERGED_DIR OUT_DIR LLAMA_CPP [QUANT]
#   MERGED_DIR  model saved by an <example>_inference.py script (or
#               save_model("merged_16bit")): weights, config, tokenizer
#               and chat_template.jinja
#   OUT_DIR     receives <name>-bf16.gguf and <name>-<QUANT>.gguf
#   LLAMA_CPP   llama.cpp checkout; requires its convert_hf_to_gguf.py
#               and a built build/bin/llama-quantize
#   QUANT       llama-quantize type, e.g. Q4_K_M (default), Q8_0
# Environment:
#   PYTHON      Python that runs convert_hf_to_gguf.py (default: python3)
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '2,14s/^# \{0,1\}//p' "$0" >&2
    exit 2
fi
merged_dir=${1%/}
out_dir=$2
llama_cpp=${3%/}
quant=${4:-Q4_K_M}
python=${PYTHON:-python3}
name=$(basename "$merged_dir")
bf16="$out_dir/$name-bf16.gguf"
quantized="$out_dir/$name-$quant.gguf"

mkdir -p "$out_dir"
# Convert once at the model's own precision; every quantization starts
# from this file. The converter also stores the tokenizer and the chat
# template, so the GGUF needs no other files.
"$python" "$llama_cpp/convert_hf_to_gguf.py" "$merged_dir" \
    --outtype bf16 --outfile "$bf16"
"$llama_cpp/build/bin/llama-quantize" "$bf16" "$quantized" "$quant"

cat <<EOF

Wrote $bf16
      $quantized

Serve (OpenAI-compatible API; --jinja applies the stored chat template,
-ngl 99 puts every layer on the GPU, -ngl 0 runs on the CPU):
  $llama_cpp/build/bin/llama-server -m $quantized --jinja -ngl 99 -c 4096 --port 8080

Ollama: a Modelfile containing the line
  FROM $(realpath "$quantized")
then: ollama create $name -f Modelfile && ollama run $name
If 'ollama show $name --modelfile' prints 'TEMPLATE {{ .Prompt }}',
Ollama did not translate the chat template: add a TEMPLATE for it.
EOF
