"""
conftest.py — mock vllm and torch for environments where they are not installed
(development / CI on non-GPU machines).
"""
import sys
from unittest.mock import MagicMock

# ---- vllm mock ----
vllm_mock = MagicMock()

class _SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=512, **kwargs):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

vllm_mock.LLM = MagicMock
vllm_mock.SamplingParams = _SamplingParams

sys.modules.setdefault("vllm", vllm_mock)

# ---- torch mock (only if not already present) ----
if "torch" not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    sys.modules["torch"] = torch_mock
