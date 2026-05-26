# Data Format

This repository keeps prompt inputs and experiment outputs separated by stage.

## Inputs

- `data/prompts/` contains the pinned Arena-Hard-v2 prompt set used by the study.
- `data/conditions/` contains prompt templates for each experimental condition.

## Outputs

- `results/behavioral/` stores raw behavioral outputs.
- `results/mechanistic/` stores patching and ablation outputs.
- `results/manipulation_check/` stores justification-check outputs.
- `results/stats/` stores bootstrap and regression summaries.
- `results/summaries/` stores aggregated CSVs used for analysis and plotting.