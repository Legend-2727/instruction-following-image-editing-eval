"""prompt_features.py — Extract interpretable features from edit prompts.

Used by ``analyze_failures.py`` to correlate prompt properties with
instruction-following failures.
"""

from __future__ import annotations

import re
from typing import Dict

# ── Keyword lists ────────────────────────────────────────────────────────────
_SPATIAL_WORDS = re.compile(
    r"\b(left|right|above|below|top|bottom|behind|in front of|next to|between|"
    r"beside|center|corner|middle|foreground|background)\b",
    re.IGNORECASE,
)

_COLORS = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|"
    r"brown|gold|silver|cyan|magenta|beige|teal|maroon|navy|violet|"
    r"turquoise|ivory|coral|lavender)\b",
    re.IGNORECASE,
)

_SPECIFICITY_PHRASES = re.compile(
    r"\b(the man|the woman|the person|the child|the dog|the cat|the car|"
    r"the building|the tree|the table|exactly|precisely|slightly|"
    r"left|right|behind|next to|on top of|underneath)\b",
    re.IGNORECASE,
)


# ── Main extractor ───────────────────────────────────────────────────────────
def extract_prompt_features(prompt: str) -> Dict[str, float]:
    """Return a dict of numeric features derived from *prompt*.

    Features
    --------
    word_count        : int   — number of whitespace-delimited tokens
    char_count        : int   — length of the raw string
    has_spatial_words : int   — 1 if any spatial keyword is present, else 0
    count_of_colors   : int   — number of color-word mentions
    num_changes_proxy : int   — count of "and" + commas (proxy for multi-edit)
    specificity_proxy : int   — count of specificity phrases (named objects,
                                directional cues, precision adverbs)
    """
    word_count = len(prompt.split())
    char_count = len(prompt)
    has_spatial = int(bool(_SPATIAL_WORDS.search(prompt)))
    color_count = len(_COLORS.findall(prompt))
    num_changes = prompt.lower().count(" and ") + prompt.count(",")
    specificity = len(_SPECIFICITY_PHRASES.findall(prompt))

    return {
        "word_count": word_count,
        "char_count": char_count,
        "has_spatial_words": has_spatial,
        "count_of_colors": color_count,
        "num_changes_proxy": num_changes,
        "specificity_proxy": specificity,
    }


PROMPT_FEATURE_NAMES = list(extract_prompt_features("").keys())
