# Data

## `prompts/arena_hard_v2_30_seed42.json`

30 prompts sampled from [Arena-Hard-v2.0-Preview](https://github.com/lmarena/arena-hard-auto) using a fixed stratified sampler:

- 3 subcategories × 10 prompts each: `coding`, `math`, `creative_writing`
- `seed=42` for the sampling RNG
- Prompts with >3000 characters excluded (see `arena_hard_v2_dropped_by_length.json`)

Each entry has: `prompt_id`, `uid`, `category`, `subcategory`, `prompt`, `char_length`.

The exact sample is reproducible by running `python judgebench.py prompts` from the repo root.

## `prompts/arena_hard_v2_dropped_by_length.json`

Record of prompts excluded from sampling because they exceeded 3000 characters. Kept for transparency about the sampling procedure.
