"""Modal entry point for mech-v2 stages (A100 80GB, bf16, pinned to the released GPU stack).

One-time setup (run yourself; keeps tokens out of chat and shell history of others):
    modal token new
    modal secret create huggingface HF_TOKEN=<your token>

Run a stage (from the repo root):
    modal run src/mech_v2/modal_app.py --stage s1 --model llama
    modal run src/mech_v2/modal_app.py --stage s2 --model llama --args '{"readout": "direct"}'
Fetch results:
    modal volume get silent-mech-v2 results results/mech_v2 --force
Summaries (CPU): python -m mech_v2.analysis s1 results/mech_v2/results/s1_llama.jsonl   (PYTHONPATH=src)
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE   # module is re-imported as /root/modal_app.py in the container

vol = modal.Volume.from_name("silent-mech-v2", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", extra_index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "safetensors==0.8.0",
                 "huggingface_hub==1.18.0", "tokenizers==0.22.2", "numpy==2.5.4", "jinja2==3.1.6")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_file(ROOT / "data/mech_v2/splits_v1.json", "/root/splits_v1.json")
    .add_local_dir(HERE, "/root/mech_v2")
)
app = modal.App("silent-mech-v2", image=image)


@app.function(gpu="A100-80GB", volumes={"/vol": vol}, timeout=4 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def run(stage: str, model: str, args: dict):
    import sys
    sys.path.insert(0, "/root")
    from mech_v2 import stages
    cx = stages.Ctx(model, "/root/splits_v1.json", "/vol/results", vol.commit,
                    attn="eager" if stage == "s4" else "sdpa")
    stages.STAGES[stage](cx, args)
    vol.commit()
    return f"{stage} {model} done"


@app.local_entrypoint()
def main(stage: str, model: str = "llama", args: str = "{}"):
    print(run.remote(stage, model, json.loads(args)))
