# Preferences: DPO of OLMo 2 1B with PEDPO

Direct Preference Optimization (DPO) teaches an instruction-tuned model
which of two responses people (or a judge model) prefer, from pairs of
`chosen` and `rejected` responses to the same prompt. It needs no reward
model and no generation during training.

This example reruns one step of a published pipeline with LoRA: AllenAI
trained [`allenai/OLMo-2-0425-1B-DPO`](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO)
from [`allenai/OLMo-2-0425-1B-SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT)
with full-model DPO on
[`allenai/olmo-2-0425-1b-preference-mix`](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix).
`prefs_train` trains a rank-1 LoRA adapter on 10,000 of those pairs with
the `PEDPO` defaults; `prefs_eval` compares the SFT model, the adapter
and AllenAI's DPO model without generating text and without a judge
model.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `prefs_eval.py` | Defines the data split (`load_preference_splits`); scores held-out preference accuracy and RewardBench implicit-reward accuracy | none (transformers + peft) |
| `prefs_train.py` | LoRA DPO | `build_config` fills a `PEDPOConfig`; `main` calls `PEDPO(config)`, `load_model()`, `train(train, held_out)`, `save_model()` |
| `prefs_inference.py` | Replies from the SFT base, the adapter and the merged model, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft, or vLLM) |

## Data

[`allenai/olmo-2-0425-1b-preference-mix`](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix):
378,301 pairs, not gated. Licence ODC-BY-1.0; it collects several
sources, some of them under non-commercial terms, and AllenAI releases
it as a research artifact. Check the dataset card before any other use.

Columns used: `chosen` and `rejected`, each a full conversation (list of
`{"role", "content"}` messages) with the same user turn(s) first and a
different final assistant turn. There is no `prompt` column: TRL takes
the shared leading turns as the prompt
([DPO data contract](../../docs/guide.md#dpo-data-contract)).

`load_preference_splits` (seed 3407):

1. drops the 1,661 pairs whose chosen and rejected conversations are
   identical (no preference);
2. shuffles the rest and holds out the first 1,000 pairs (never
   trained on);
3. trains on the next 10,000 pairs.

1 of the 1,000 held-out prompts also occurs in the training pairs (with
other responses).

`PEDPO` renders both conversations with the OLMo 2 chat template and
truncates prompt + response at `max_length` (1024 tokens). TRL drops a
pair when the prompt alone fills `max_length`: 475 of the 10,000
training pairs (9,525 remain, 596 steps of 16 pairs) and 62 of the 1,000
held-out pairs in TRL's eval metrics. In 27% of the training pairs at
least one of the two sequences is longer than 1024 tokens and loses the
end of its response.

One pair as `prefs_train` prints it (prompt with the assistant header,
then the two responses; shortened; the `&#39;` entities are in the
data):

```text
<|endoftext|><|user|>
Refine the subsequent Python code:

def unique_pairs(lst1, lst2):
    &#39;&#39;&#39;
    Construct a function that accepts two lists of strings as input. [...]
    &#39;&#39;&#39;
<|assistant|>

--- chosen ---
To accomplish the provided requirements, we first should preprocess the lists by ignoring case and any repeated strings. [...]
--- rejected ---
Here is a refined version of your Python function that follows the problem statement: [...]
```

## Model

`allenai/OLMo-2-0425-1B-SFT`, Apache-2.0, 1.48B parameters, trained in
bf16 (no quantization). Its chat template puts `<|user|>\n` and
`<|assistant|>\n` before each turn and `<|endoftext|>` after each
assistant turn; the response tokens DPO
scores are the assistant text plus that `<|endoftext|>`.

## Rank check

```bash
uv run python -m speftr.lora_budget --mode dpo \
    --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --max_steps 596 --pairs_per_step 16
```

```text
Adapter params:  753,664 (rank 1)
Pairs:           9,536 (1 bit per pair)
Required params: 4,768 (estimate: bits / 2)
Minimum rank:    1 (estimate); rank 1 suffices
```

A pair carries about one bit, so 10,000 pairs need about 5,000
parameters; rank 1 on every attention and MLP projection has 753,664.
Rank 1 has room for the whole mix (~189k parameters needed,
[guide](../../docs/guide.md#rank)).

## Run

From the repository root:

```bash
# 1. Train (the documented run)
uv run python -m examples.prefs.prefs_train
# 2. Evaluate SFT, adapter and AllenAI's DPO model
uv run python -m examples.prefs.prefs_eval
```

Main flags (`--help` lists all):

| Script | Flag | Default |
|--------|------|---------|
| `prefs_train` | `--model_name_or_path` | `allenai/OLMo-2-0425-1B-SFT` |
| | `--train_pairs`, `--eval_pairs` | 10,000, 1,000 |
| | `--max_steps` | -1 (one epoch) |
| | `--loss_type`, `--beta` | `sigmoid`, 0.1 |
| | `--precompute_ref_log_probs` | off |
| | `--output_dir` | `./models/olmo2-1b-lora-dpo` |
| `prefs_eval` | `--adapter_dir` | `./models/olmo2-1b-lora-dpo` |
| | `--reference_model` | `allenai/OLMo-2-0425-1B-DPO` (`""` skips it) |
| | `--eval_pairs` | 1,000 (must match training) |
| | `--rewardbench_rows` | -1 (all 2,985) |
| | `--max_length`, `--token_budget` | 4096, 16384 |

Everything else is a `PEDPO` default: rank 1, alpha 32, learning rate
5e-6 constant, `beta` 0.1, sigmoid loss, 4 pairs × 4 accumulation steps,
gradient checkpointing, `adamw_8bit`.

## Results

One RTX 3090, bf16, the `PEDPO` defaults, 9,525 pairs, one epoch (596
steps). `prefs_eval` on the same GPU.

### Training

| Wall time | Training | s/step | Final eval | Peak memory (torch) |
|-----------|----------|--------|------------|---------------------|
| 102 min | 96.6 min | 9.73 | 4.5 min | 12.7 GB allocated, 15.0 GB reserved |

"Wall time" is the whole `PEDPO.train` call (tokenization, training,
final evaluation); `prefs_train_summary.json` records it.

| Metric | Value |
|--------|-------|
| Mean train loss (epoch) | 0.648 |
| Last logged train loss (steps 581-590) | 0.616 |
| Last logged train `rewards/accuracies` | 0.713 |
| `eval_loss` (938 held-out pairs) | 0.638 |
| `eval_rewards/accuracies` | 0.635 |
| `eval_rewards/margins` | 0.270 |

The loss starts at 0.693 (ln 2, policy = reference) and train
`rewards/accuracies` rises from 0.44 in the first 10 steps to 0.6-0.7
from about step 130 on.

### Scores

About 10 minutes for all three models, 11.0 GB peak (torch).

Held-out preference accuracy (1,000 pairs, never trained on):

| Model | Log-likelihood, summed | Log-likelihood, per token | Implicit reward, summed | Implicit reward, per token |
|-------|------------------------|---------------------------|-------------------------|----------------------------|
| `OLMo-2-0425-1B-SFT` | 0.514 | 0.617 | | |
| SFT + DPO adapter (this example) | 0.515 | 0.621 | 0.635 | 0.653 |
| `OLMo-2-0425-1B-DPO` (AllenAI, full model, whole mix) | 0.551 | 0.667 | 0.695 | 0.770 |

RewardBench implicit-reward accuracy (`filtered`, 2,985 pairs; reference
= SFT model):

| Policy | Reward | Chat | Chat Hard | Safety | Reasoning | Overall |
|--------|--------|------|-----------|--------|-----------|---------|
| SFT + DPO adapter (this example) | summed | 0.791 | 0.557 | 0.691 | 0.767 | 0.701 |
| `OLMo-2-0425-1B-DPO` (AllenAI) | summed | 0.494 | 0.695 | 0.695 | 0.776 | 0.665 |
| SFT + DPO adapter (this example) | per token | 0.869 | 0.388 | 0.631 | 0.513 | 0.600 |
| `OLMo-2-0425-1B-DPO` (AllenAI) | per token | 0.880 | 0.447 | 0.730 | 0.573 | 0.658 |

- The adapter learns the preferences in DPO's own terms: its implicit
  reward ranks 63.5% of the held-out pairs correctly, the same as TRL's
  `eval_rewards/accuracies` at 1024 tokens, and 70.1% of RewardBench
  overall.
- It barely changes which response the model itself finds more likely
  (0.514 to 0.515 summed, 0.617 to 0.621 per token); AllenAI's DPO
  model moves them by 3.7 and 5.0 points. See [below](#why-the-likelihood-accuracy-barely-moves).
- On RewardBench the adapter's reward is higher on Chat (0.791 vs
  0.494) and lower on Chat Hard (0.557 vs 0.695) than AllenAI's; Safety
  and Reasoning are within 0.01.

### Why the likelihood accuracy barely moves

Checked on the 1,000 held-out pairs:

- **Formatting and masking.** `prefs_eval` tokenizes as TRL does
  (tested in `tests/test_examples_prefs.py`), and its implicit-reward
  accuracy (0.635 on 1,000 pairs at 4096 tokens) matches TRL's (0.635
  on the 938 pairs that fit in 1024).
- **Size of the change.** Under the SFT model, the summed
  log-probabilities of the chosen and rejected response differ by a
  median of 201 nats (mean 539): the two responses differ in length and
  content. The adapter widens the chosen-minus-rejected gap by a median
  of 1.3 nats (mean 3.1) and flips 4 pairs to correct and 3 to wrong.
  AllenAI's model widens it by a median of 16 nats (mean 60) and flips
  42 to correct and 5 to wrong.
- **Learning rate.** A 200-step run at 5e-5 (10x the default) reached
  the same `eval_rewards/accuracies` (0.636) and a median gap change of
  2.0 nats; likelihood accuracy 0.519.

So the adapter is trained correctly and generalizes, but 9,525 pairs
with a rank-1 adapter shift the log-probabilities by a few nats, far
less than the gaps that decide a likelihood comparison. AllenAI's model
was trained on all 378k pairs with every weight free. The defaults stay
as they are; to move likelihood rankings, train on more pairs.

### Other settings measured

Same data, split and seed; one setting changed each time.

- Length-normalized DPO, the loss of AllenAI's Tülu 3 recipe
  ([arXiv 2411.15124](https://arxiv.org/abs/2411.15124), Table 20):
  `--loss_type sigmoid_norm --beta 5`. Held-out per-token
  implicit-reward accuracy is 0.650 (default 0.653) and RewardBench
  overall 0.662 (default 0.701). The loss does not explain the gap to
  AllenAI's model, which trains all parameters on the whole mix
  (~378k pairs, ~40x more); neither factor is tested here.
- `--precompute_ref_log_probs`: 7.5 s/step (default 9.7) plus a
  25-minute reference pass before training, so one epoch takes the same
  total time (103 vs 102 min). It pays off when the same pairs are
  trained for several epochs. Peak memory is unchanged (12.5 vs
  12.7 GB).

## Evaluation

`prefs_eval` scores every response by its summed token log-probability
under a model, in padded batches sorted by length (`--token_budget`
tokens per forward pass). Prompts and responses are tokenized as TRL
does in training (`encode_response`), and sequences are cut at 4096
tokens (the model's context), not at the training `max_length`.

- **Held-out preference accuracy.** On the 1,000 held-out pairs: the
  fraction where the model gives the chosen response a higher
  log-probability than the rejected one, summed over tokens ("summed")
  and divided by the response length ("per token"). "Implicit reward" is the
  implicit-reward accuracy below on the same pairs, the quantity DPO
  optimizes; TRL's `eval_rewards/accuracies` is the summed measure on
  the pairs that fit in 1024 tokens.
- **RewardBench implicit-reward accuracy.**
  [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench)
  (ODC-BY), `filtered` split, 2,985 prompts with a chosen and a rejected
  response in 23 subsets. A DPO model defines a reward
  `beta * (log pi(y|x) - log pi_ref(y|x))`, summed over the response
  tokens, with the SFT model as `pi_ref`; a pair is correct when the
  chosen response gets the higher reward (ties count as wrong). For the
  adapter, `pi` is the model with the adapter and `pi_ref` the same
  model with it disabled; for AllenAI's DPO model, `pi` is that model
  and `pi_ref` the SFT model. The "per token" reward divides the
  log-ratio by the number of response tokens, the reward that
  `sigmoid_norm` optimizes. Each prompt is one user turn rendered with
  the chat template. Sections and weights follow RewardBench
  (`rewardbench/constants.py`): Chat, Chat Hard and Safety weight each
  subset by its size; in Reasoning, `math-prm` counts as much as the six
  `hep-*` code subsets together. "Overall" is the mean of the four
  sections (RewardBench's leaderboard also mixes in prior preference
  sets for DPO models, which this example leaves out).

## Outputs

`--output_dir` receives `adapter_model.safetensors`, `adapter_config.json`
(records the base model), tokenizer files with `chat_template.jinja`,
`prefs_train_summary.json` (time, memory, last train log, final eval
metrics) and a `checkpoint-*` directory. `prefs_eval` writes
`prefs_eval.json` next to them. To load, merge or serve the adapter, see
[After training](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`prefs_inference.py` answers a few prompts with only torch,
transformers, peft and (for `--engine vllm`) vllm. It runs three routes
and prints each reply:

- **base**: the SFT model without the adapter (`run_base_route`);
- **adapter**: the SFT model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`; vLLM
  supports LoRA on a bf16 base);
- **merged**: the adapter merged into the bf16 base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.prefs.prefs_inference
uv run python -m examples.prefs.prefs_inference --engine vllm \
    --prompt "Give me three tips for writing a clear email."
```

Output of the first command (RTX 3090, about a minute, `max_new_tokens`
200; the merged replies, equal to the adapter's, shortened):

```text
Engine: transformers. Merged model saved to ./models/olmo2-1b-lora-dpo-merged

>>> Write a haiku about autumn leaves.
--- base ---
Red, orange, yellow
Leaves fall from the trees
Nature's beauty
--- adapter ---
Red, orange, yellow
Leaves fall from the trees
Nature's beauty
--- merged ---
Red, orange, yellow
Leaves fall from the trees
Nature's beauty

>>> How can I get better sleep? Answer in three short points.
--- base ---
1. Establish a regular sleep schedule. Try to go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's internal clock and improve sleep quality.
2. Create a restful environment. Make your bedroom dark, quiet, and cool. Consider using blackout curtains, earplugs, or a fan to create a comfortable sleep environment.
3. Limit exposure to screens before bed. The blue light emitted by screens can interfere with your ability to fall asleep. Try to avoid screens at least an hour before bed.
--- adapter ---
1. Establish a regular sleep schedule by going to bed and waking up at the same time every day, even on weekends.
2. Create a sleep-friendly environment by keeping your bedroom cool, dark, and quiet, and avoiding screens before bed.
3. Limit caffeine and alcohol intake, as they can interfere with sleep quality.
--- merged ---
(same as adapter)
```

With `--engine vllm` (about 3.5 minutes, one engine per route) the
adapter and merged replies are identical to the transformers ones; the
base reply to the sleep prompt differs from the transformers base reply
after its first point.

`./models/olmo2-1b-lora-dpo-merged` (default `<adapter_dir>-merged`)
holds the merged bf16 weights, config, tokenizer and
`chat_template.jinja`; it loads without peft, like any Hub checkpoint.

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your data

1. **Pairs.** Replace `load_preference_splits` with a loader that
   returns `datasets.Dataset` objects with `chosen` and `rejected`
   conversations, or with `prompt`, `chosen` and `rejected` columns
   ([DPO data contract](../../docs/guide.md#dpo-data-contract)). Keep a
   held-out split that training never sees. The evaluation encoders
   (`encode_conversation_pairs`, `encode_rewardbench`) show both layouts.
2. **Model.** Pass `--model_name_or_path` with an SFT model that has a
   chat template (or a merged SFT checkpoint of your own). The reference
   is that model; set `--reference_model` in `prefs_eval` to a released
   DPO model of the same family, or `""`.
3. **Length.** Check the token length of your pairs: with
   `max_length` 1024, prompts that fill it are dropped and long
   responses are cut. Raise `max_length` in `build_config` if memory
   allows (activation memory grows with it).
4. **Rank.** Rerun the [rank check](#rank-check) with your pair count.

## Pitfalls

- Pairs with identical `chosen` and `rejected` carry no signal; drop
  them (`split_pairs` does).
- A response that starts with a newline merges with the `\n` of the
  assistant header into one token, so the tokenized prompt is not a
  prefix of prompt + response. TRL then warns "Mismatch between
  tokenized prompt and the start of tokenized prompt+chosen" (or
  "+rejected") and
  splits at the prompt's token count anyway; 96 of the 22,000
  sequences here are affected. `prefs_eval` splits the same way.
- `precompute_ref_log_probs` caches the reference log-probs in the
  `datasets` cache; a run that finds a matching cache file skips the
  pass, so time it with a clean cache.
- Summed log-likelihood favours short responses; compare it with the
  per-token accuracy before reading a change as a preference change.
