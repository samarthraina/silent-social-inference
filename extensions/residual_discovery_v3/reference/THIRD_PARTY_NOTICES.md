# Third-party material

The root [MIT License](LICENSE) applies to original project code and documentation owned
by Samarth Raina. It does not replace the terms of included third-party source, benchmark
text, or model-generated research records. No endorsement by upstream projects is implied.

## Jacobian Lens

The reference implementation is vendored in `third_party/jacobian-lens/` at revision
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. Its [Apache-2.0 license](third_party/jacobian-lens/LICENSE)
and upstream notices are retained. See [the vendoring record](third_party/jacobian-lens/VENDORING.md)
for provenance. The pretrained fitted tensor is a separate artifact and is not included.

## Benchmark inputs

The following notices apply to benchmark text wherever it appears in this repository,
including pooled task files, prompts embedded in result records, and quoted examples.
The dataset identity and source task ID identify the applicable source.

| Benchmark and source attribution | Pinned revision | Terms |
|---|---|---|
| MT-Bench, LMSYS, distributed by HuggingFaceH4 | `e3a795c5e9a82ee40611c416b8a7786c73198991` | [Apache-2.0 dataset card](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/blob/e3a795c5e9a82ee40611c416b8a7786c73198991/README.md) |
| Arena-Hard v2, LMArena | `196f6b826783b3da7310e361a805fa36f0be83f3` | [Apache-2.0 license](https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/LICENSE) |
| WildBench v2, Allen Institute for AI and WildBench contributors | `26c49eb39d7d5ce2099b0bbafed5a88dcce954ec` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [source dataset card](https://huggingface.co/datasets/allenai/WildBench/blob/26c49eb39d7d5ce2099b0bbafed5a88dcce954ec/README.md) |
| AlpacaEval, Stanford Tatsu Lab and dataset contributors | `2edc6fad8be6b14ea7230aabfd08188da6b8b814` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), [source dataset card](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/2edc6fad8be6b14ea7230aabfd08188da6b8b814/README.md) |

A full copy of the standard [Apache-2.0 license](third_party/jacobian-lens/LICENSE) is
included. Creative Commons license links above identify the applicable legal terms.
AlpacaEval appears in earlier-stage task pools and is not part of the final balanced
three-dataset study. Its noncommercial condition continues to apply to that benchmark
material, including copies inside combined files; it does not change the MIT license on
original project code.

Processing consists of task selection, structural filtering, first-turn extraction where
applicable, ID preservation, JSON reformatting, and joining task text to generated answers
and evaluation records. These are adapted distributions of the benchmark inputs. Exact
source metadata and selection rules are retained in `code/data/prompts/` and the
`code/scripts/prepare_*` scripts. Two credential-like strings in an earlier WildBench task
are redacted in four prompt pools, as recorded in [PUBLIC_REDACTIONS.json](PUBLIC_REDACTIONS.json).
The final scoring inputs are unchanged. This release does not claim authorship of benchmark questions or transfer rights that
the original sources do not grant.

## Models and generated records

No model weights are included. Model identifiers and pinned revisions are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Downloading or using each model is subject to its
own terms, including gated access where applicable. Retained model outputs are part of the
research records and are not represented as independently authored reference answers or
relicensed wholesale under MIT. Any benchmark text reproduced in an output remains subject
to the benchmark notices above.

Model sources are [OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct),
[OLMo 3.1 32B](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct),
[Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B),
[Gemma 2 9B](https://huggingface.co/google/gemma-2-9b-it),
[Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and
[gpt-oss 20B](https://huggingface.co/openai/gpt-oss-20b).
Gemma distinguishes generated output from model derivatives and does not claim rights in
generated outputs; see [section 3.3 of its terms](https://ai.google.dev/gemma/terms).
The [Llama 3.1 community license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
contains additional conditions for using its outputs to train or improve models.
This repository releases experimental records, not model weights or a trained model.

## Dependencies

Analysis, testing, and execution dependencies are installed separately. Their respective
licenses continue to apply. The production environment is historical provenance; the CPU
quick start does not claim to reproduce that GPU environment.
