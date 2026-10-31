# Release checks (2026-10-31)

- All planned raw units, payload hashes, frozen split inputs, executed-source hashes and candidate hashes verified from the packaged files.
- Both original analyzer summaries independently recomputed from raw records, including bootstrap intervals, and matched all saved fields to 1e-12.
- 20 scientific unit/architecture tests passed with the packaged reference tree, HF_HUB_OFFLINE=1, torch 2.15.0 and transformers 5.22.0.
- The two-page PDF was rendered and both pages visually inspected.
- Publication scan found no Hugging Face/GitHub access tokens, private keys, private local home paths or server connection addresses in the extension. Benchmark questions and generated evaluation text retain their original bytes and upstream terms.

No new GPU inference or held-out validation was performed while preparing this release.
