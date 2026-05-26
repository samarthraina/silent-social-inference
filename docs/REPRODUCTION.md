# Reproduction

This project is organized so the full pipeline can be rerun from the tracked
inputs in `data/`, `configs/`, `src/`, and `scripts/`.

## Suggested flow

1. Install dependencies from `requirements.txt`.
2. Run the one-shot scripts in `scripts/` or the individual modules in `src/`.
3. Regenerate summary tables in `results/stats/` and plots in `results/figures/`.
4. Copy final publication figures into `figures/`.

See the root `README.md` for the current high-level pipeline description.