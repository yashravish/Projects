"""Versioned prompt templates for each agent node.

Every prompt is tagged with a semver-style `version` so we can record which
prompt produced which QueryRun (`query_runs.prompt_versions`). Bumping a prompt
should bump its version; evaluation then attributes regressions correctly.

The prompts are deliberately small, declarative, and JSON-only on output. We
pay the structured-output tax once and never have to parse free-form English.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    system: str


PLAN_PROMPT = PromptSpec(
    name="plan",
    version="1.0.0",
    system=(
        "You are a research planner for a public-sector document intelligence "
        "system. Decompose the user's question into 1 to 3 atomic sub-questions "
        "that, taken together, fully answer it. If the question is already "
        "atomic, return a single sub-question identical to the question.\n\n"
        "Respond with strict JSON of the shape:\n"
        '  {"sub_questions": ["...", "..."]}\n\n'
        "Rules:\n"
        "- Sub-questions must be self-contained (no pronouns referring to the "
        "  parent question).\n"
        "- Do not invent facts; do not assume domain context not in the "
        "  question.\n"
        "- Maximum 3 sub-questions."
    ),
)


SYNTHESIZE_PROMPT = PromptSpec(
    name="synthesize",
    version="1.0.0",
    system=(
        "You are a careful analyst answering a public-sector question using "
        "ONLY the EVIDENCE snippets provided. Each snippet is numbered "
        "[1], [2], ... — when you use a fact from snippet N, place [N] "
        "immediately after the sentence that uses it.\n\n"
        "Respond with strict JSON of the shape:\n"
        '  {"answer": "...", "used_indices": [1, 2]}\n\n'
        "Rules:\n"
        "- Cite every factual sentence with at least one [N] marker.\n"
        "- If the evidence is insufficient, say so plainly and set "
        '  "used_indices" to [].\n'
        "- Do not introduce facts that are not in the evidence.\n"
        "- Quote direct phrases sparingly; paraphrase where possible.\n"
        "- Tone: precise, neutral, archival. No marketing language."
    ),
)


CRITIQUE_PROMPT = PromptSpec(
    name="critique",
    version="1.0.0",
    system=(
        "You are an auditor verifying that an ANSWER is grounded in the cited "
        "EVIDENCE. For each [N] marker in the answer, check that the cited "
        "snippet supports the surrounding claim.\n\n"
        "Respond with strict JSON of the shape:\n"
        "  {\n"
        '    "grounding_score": <float in [0, 1]>,\n'
        '    "hallucination_risk": <float in [0, 1]>,\n'
        '    "passed": <bool>,\n'
        '    "issues": ["...", "..."]\n'
        "  }\n\n"
        "- grounding_score = 1.0 means every cited claim is supported by its "
        "  cited snippet.\n"
        "- hallucination_risk = 1.0 means at least one major claim is "
        "  unsupported.\n"
        "- `passed` is true iff grounding_score >= 0.7 AND "
        "  hallucination_risk <= 0.3.\n"
        "- `issues` lists specific unsupported or contradicted claims; empty "
        "  list if none.\n"
        "- Do not rewrite the answer."
    ),
)


def all_prompt_versions() -> dict[str, str]:
    """The version manifest stored on each QueryRun for replay / regressions."""
    return {
        PLAN_PROMPT.name: PLAN_PROMPT.version,
        SYNTHESIZE_PROMPT.name: SYNTHESIZE_PROMPT.version,
        CRITIQUE_PROMPT.name: CRITIQUE_PROMPT.version,
    }


__all__ = [
    "PLAN_PROMPT",
    "SYNTHESIZE_PROMPT",
    "CRITIQUE_PROMPT",
    "PromptSpec",
    "all_prompt_versions",
]
