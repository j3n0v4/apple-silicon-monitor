"""Standardized benchmark prompts for LLM inference benchmarking.

Each prompt is designed to exercise different aspects of model performance:
- short: Minimal generation, tests prompt processing speed
- medium: Moderate generation, tests sustained throughput
- long: Extended generation, tests memory bandwidth and long-context handling
"""

from typing import Literal

PromptLength = Literal["short", "medium", "long"]

PROMPTS: dict[PromptLength, str] = {
    "short": "Explain quantum computing in 3 sentences.",
    "medium": (
        "Write a Python function that finds the longest palindromic substring "
        "in a given string. Include edge case handling and a brief explanation "
        "of the algorithm."
    ),
    "long": (
        "Write a detailed comparison of transformer architectures versus "
        "state-space models (like Mamba). Cover: (1) computational complexity, "
        "(2) memory requirements, (3) training efficiency, (4) inference speed, "
        "(5) suitability for different task types. Provide at least 3 concrete "
        "examples for each."
    ),
}

ALL_LENGTHS: list[PromptLength] = ["short", "medium", "long"]
