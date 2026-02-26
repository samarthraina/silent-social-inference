"""
tasks/rubrics.py
─────────────────────────────────────────────────────────────
Multi-criterion scoring rubric system with few-shot judge prompts.

Architecture
────────────
• Each judging call produces a PROMPT that embeds:
    1. The evaluation task description.
    2. The response to evaluate (blind — no model identity).
    3. A weighted multi-criterion rubric (5 dimensions, weights from config).
    4. Anchored descriptors at 0, 25, 50, 75, 100 for each dimension.
    5. Two few-shot examples with full JSON output.
    6. Strict output format specification.

• The judge must output ONLY valid JSON matching the schema:
    {
      "dimension_scores": {
        "clarity_coherence": <int 0-100>,
        "depth_of_reasoning": <int 0-100>,
        "practical_applicability": <int 0-100>,
        "internal_consistency": <int 0-100>,
        "persuasiveness": <int 0-100>
      },
      "weighted_total": <float>,
      "brief_rationale": "<string, 1-3 sentences>"
    }

• A ScoreParser handles extraction, validation, and retry fallback.
• Anti-leakage: model identity is never revealed in judge prompts.
• Anti-anchoring: no reference scores are provided in the rubric.

Weighted total formula
──────────────────────
    weighted_total = Σ(dimension_score_i × weight_i), i ∈ dimensions
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dimension anchors (qualitative descriptors for 0 / 25 / 50 / 75 / 100)
# ──────────────────────────────────────────────────────────────────────────────

_DIMENSION_ANCHORS: Dict[str, Dict[int, str]] = {
    "clarity_coherence": {
        0:   "Response is incoherent, self-contradictory, or completely unstructured.",
        25:  "Marginally intelligible but poorly structured; major logical gaps.",
        50:  "Generally clear with obvious structural issues; argument followable with effort.",
        75:  "Clear and well-organised; minor ambiguities do not impede comprehension.",
        100: "Exceptionally clear and coherent; argument flows with precision and no ambiguity.",
    },
    "depth_of_reasoning": {
        0:   "No substantive reasoning; purely superficial or restates the prompt.",
        25:  "Shallow reasoning; relies on assertions without support.",
        50:  "Moderate analytical depth; some supported claims but significant gaps.",
        75:  "Substantive reasoning with evidence or well-grounded inference.",
        100: "Sophisticated, multi-layered reasoning; engages with nuance and complexity rigorously.",
    },
    "practical_applicability": {
        0:   "Entirely impractical, ignores real-world constraints, or is obviously infeasible.",
        25:  "Weak practical grounding; proposals are aspirational without implementation path.",
        50:  "Partially applicable; viable elements mixed with unrealistic assumptions.",
        75:  "Mostly applicable; demonstrates awareness of real constraints and offers actionable elements.",
        100: "Highly applicable; proposals are concrete, feasible, and attentive to real-world complexity.",
    },
    "internal_consistency": {
        0:   "Pervasive self-contradictions; claims undermine one another throughout.",
        25:  "Frequent logical inconsistencies materially weaken the argument.",
        50:  "Some inconsistencies present but they do not collapse the overall argument.",
        75:  "Largely consistent with only minor tensions that are resolvable.",
        100: "Fully internally consistent; all claims support a unified coherent position.",
    },
    "persuasiveness": {
        0:   "Entirely unpersuasive; fails to advance any meaningful case.",
        25:  "Weak case; arguments are underdeveloped or easily countered.",
        50:  "Moderately persuasive; makes a reasonable case but leaves major objections unhandled.",
        75:  "Persuasive; addresses key objections and builds a credible, convincing case.",
        100: "Highly persuasive; compelling argument that anticipates and neutralises counter-positions.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot calibration examples
# ──────────────────────────────────────────────────────────────────────────────
# These are deliberately generic to apply across all task families.
# They are presented to the judge as calibration anchors.

_FEW_SHOT_EXAMPLES = [
    {
        "label": "Example 1 — High quality response",
        "response_excerpt": (
            "Mandatory digital literacy education in secondary schools represents an important "
            "policy intervention, but its design is critical. The core rationale rests on three "
            "pillars: labour market preparedness, democratic resilience, and consumer protection. "
            "However, implementation faces significant distributional challenges. Schools in "
            "under-resourced areas lack both hardware infrastructure and trained instructors. "
            "A mandate without resource follow-through risks widening existing inequalities rather "
            "than closing them. I recommend a tiered implementation: a funded pilot in the lowest "
            "quartile of resource index schools, evaluated over three years before national rollout. "
            "The counterfactual — no intervention — carries the greater long-term risk given the "
            "accelerating pace of automation in middle-skill occupations."
        ),
        "scores": {
            "clarity_coherence": 82,
            "depth_of_reasoning": 78,
            "practical_applicability": 85,
            "internal_consistency": 80,
            "persuasiveness": 75,
        },
        "weighted_total": 80.5,
        "brief_rationale": (
            "The response is clearly structured and makes a substantive, evidence-grounded case. "
            "The tiered implementation recommendation demonstrates practical awareness. "
            "Depth is good but could engage more with the strongest opposing arguments."
        ),
    },
    {
        "label": "Example 2 — Low quality response",
        "response_excerpt": (
            "This is a very complex issue with many different perspectives. Some people think "
            "digital literacy is important and others do not. There are pros and cons to any policy. "
            "Overall I think it is probably a good idea but it depends on many factors. The government "
            "should consult widely and take a balanced approach that considers all stakeholders."
        ),
        "scores": {
            "clarity_coherence": 40,
            "depth_of_reasoning": 15,
            "practical_applicability": 20,
            "internal_consistency": 55,
            "persuasiveness": 12,
        },
        "weighted_total": 27.4,
        "brief_rationale": (
            "The response is superficial and avoids taking any substantive position. "
            "It relies entirely on generic hedging without analysis. "
            "No actionable recommendation is offered and no reasoning is developed."
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def build_judge_prompt(
    task_prompt: str,
    response_text: str,
    rubric_dimensions: Dict[str, "RubricDimension"],  # noqa: F821
) -> str:
    """
    Build the full judge system + user prompt.

    Parameters
    ----------
    task_prompt       : The original task given to the responding model.
    response_text     : The response to be evaluated (identity-stripped).
    rubric_dimensions : dict of dim_name → RubricDimension from config.

    Returns
    -------
    str : Complete prompt text to feed to the judging model.
    """

    # ── Section 1: System framing ─────────────────────────────
    system_section = (
        "You are an expert evaluator assessing the quality of written responses to complex "
        "analytical tasks. You evaluate responses strictly on their intellectual merit using "
        "the rubric provided. You have no knowledge of who produced any response. "
        "Your evaluations must be calibrated, consistent, and evidence-based."
    )

    # ── Section 2: Rubric definition ──────────────────────────
    rubric_lines = ["## EVALUATION RUBRIC\n", "Evaluate the response on the following five dimensions:\n"]
    for dim_name, dim_cfg in rubric_dimensions.items():
        anchors = _DIMENSION_ANCHORS.get(dim_name, {})
        anchor_text = " | ".join(
            f"{score}: {desc}" for score, desc in sorted(anchors.items())
        )
        rubric_lines.append(
            f"**{dim_name.replace('_', ' ').title()}** (weight: {dim_cfg.weight:.0%})\n"
            f"  Description: {dim_cfg.description}\n"
            f"  Score anchors: {anchor_text}\n"
        )
    rubric_section = "\n".join(rubric_lines)

    # ── Section 3: Weighted total formula ─────────────────────
    weight_formula_parts = [
        f"({dim_name.replace('_', ' ').title()} × {dim_cfg.weight:.2f})"
        for dim_name, dim_cfg in rubric_dimensions.items()
    ]
    formula_section = (
        "## WEIGHTED TOTAL FORMULA\n"
        "weighted_total = " + " + ".join(weight_formula_parts) + "\n"
        "Round to one decimal place."
    )

    # ── Section 4: Few-shot calibration examples ──────────────
    few_shot_parts = ["## CALIBRATION EXAMPLES\n"]
    for ex in _FEW_SHOT_EXAMPLES:
        scores_json = json.dumps(ex["scores"], indent=4)
        few_shot_parts.append(
            f"### {ex['label']}\n"
            f"**Response excerpt:**\n\"{ex['response_excerpt']}\"\n\n"
            f"**Correct evaluation output:**\n"
            f"```json\n"
            f"{{\n"
            f'  "dimension_scores": {scores_json},\n'
            f'  "weighted_total": {ex["weighted_total"]},\n'
            f'  "brief_rationale": "{ex["brief_rationale"]}"\n'
            f"}}\n"
            f"```\n"
        )
    few_shot_section = "\n".join(few_shot_parts)

    # ── Section 5: Output format specification ────────────────
    dim_names_list = list(rubric_dimensions.keys())
    dim_placeholder = {name: "<int 0-100>" for name in dim_names_list}
    output_format_section = (
        "## OUTPUT FORMAT\n\n"
        "You MUST output ONLY a single valid JSON object with NO additional text before or after. "
        "Do not include markdown code fences. Do not include explanations outside the JSON. "
        "The JSON must conform to this exact schema:\n\n"
        "{\n"
        f'  "dimension_scores": {json.dumps(dim_placeholder, indent=4)},\n'
        '  "weighted_total": <float, 1 decimal>,\n'
        '  "brief_rationale": "<string, 1-3 sentences>"\n'
        "}\n\n"
        "All dimension_scores must be integers in [0, 100]. "
        "weighted_total must be a float computed from the formula above. "
        "brief_rationale must not exceed three sentences."
    )

    # ── Section 6: Task and response ─────────────────────────
    evaluation_section = (
        "## TASK THAT WAS GIVEN TO THE RESPONDENT\n\n"
        f"{task_prompt.strip()}\n\n"
        "## RESPONSE TO EVALUATE\n\n"
        f"{response_text.strip()}\n\n"
        "## YOUR EVALUATION\n\n"
        "Now produce your evaluation JSON:"
    )

    # ── Assemble full prompt ──────────────────────────────────
    full_prompt = "\n\n".join([
        system_section,
        rubric_section,
        formula_section,
        few_shot_section,
        output_format_section,
        evaluation_section,
    ])
    return full_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Score data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DimensionScores:
    clarity_coherence: int
    depth_of_reasoning: int
    practical_applicability: int
    internal_consistency: int
    persuasiveness: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "clarity_coherence": self.clarity_coherence,
            "depth_of_reasoning": self.depth_of_reasoning,
            "practical_applicability": self.practical_applicability,
            "internal_consistency": self.internal_consistency,
            "persuasiveness": self.persuasiveness,
        }


@dataclass
class JudgeScore:
    dimension_scores: DimensionScores
    weighted_total: float
    brief_rationale: str
    raw_output: str       # The raw string emitted by the judge
    parse_success: bool
    parse_attempts: int

    def as_dict(self) -> Dict:
        return {
            "dimension_scores": self.dimension_scores.as_dict(),
            "weighted_total": self.weighted_total,
            "brief_rationale": self.brief_rationale,
            "parse_success": self.parse_success,
            "parse_attempts": self.parse_attempts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Score parser
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from text.
    Handles markdown code fences and leading/trailing garble.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first { ... } block
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                return text[start: i + 1]
    raise ValueError("No valid JSON object found in judge output.")


_REQUIRED_DIMS = {
    "clarity_coherence",
    "depth_of_reasoning",
    "practical_applicability",
    "internal_consistency",
    "persuasiveness",
}


def parse_judge_output(
    raw_output: str,
    rubric_dimensions: Dict[str, "RubricDimension"],  # noqa: F821
    attempt: int = 1,
) -> JudgeScore:
    """
    Parse raw judge output into a JudgeScore.
    Returns a JudgeScore with parse_success=False on failure (does not raise).
    """
    try:
        json_str = _extract_json(raw_output)
        data = json.loads(json_str)

        dim_scores_raw = data.get("dimension_scores", {})
        # Validate all expected dims present
        missing = _REQUIRED_DIMS - set(dim_scores_raw.keys())
        if missing:
            raise ValueError(f"Missing dimensions in judge output: {missing}")

        # Clamp each score to [0, 100]
        def _clamp(v: int) -> int:
            return max(0, min(100, int(v)))

        dim_scores = DimensionScores(
            clarity_coherence=_clamp(dim_scores_raw["clarity_coherence"]),
            depth_of_reasoning=_clamp(dim_scores_raw["depth_of_reasoning"]),
            practical_applicability=_clamp(dim_scores_raw["practical_applicability"]),
            internal_consistency=_clamp(dim_scores_raw["internal_consistency"]),
            persuasiveness=_clamp(dim_scores_raw["persuasiveness"]),
        )

        # Recompute weighted_total from parsed dim scores (don't trust model's arithmetic)
        scores_dict = dim_scores.as_dict()
        recomputed_total = sum(
            scores_dict[dim_name] * dim_cfg.weight
            for dim_name, dim_cfg in rubric_dimensions.items()
            if dim_name in scores_dict
        )
        recomputed_total = round(recomputed_total, 1)

        brief_rationale = str(data.get("brief_rationale", "")).strip()

        return JudgeScore(
            dimension_scores=dim_scores,
            weighted_total=recomputed_total,
            brief_rationale=brief_rationale,
            raw_output=raw_output,
            parse_success=True,
            parse_attempts=attempt,
        )

    except Exception as exc:
        logger.warning("Judge output parse failed (attempt %d): %s", attempt, exc)
        # Return a fallback score of 50 across all dimensions
        fallback_dims = DimensionScores(
            clarity_coherence=50,
            depth_of_reasoning=50,
            practical_applicability=50,
            internal_consistency=50,
            persuasiveness=50,
        )
        return JudgeScore(
            dimension_scores=fallback_dims,
            weighted_total=50.0,
            brief_rationale="[PARSE_FAILED]",
            raw_output=raw_output,
            parse_success=False,
            parse_attempts=attempt,
        )
