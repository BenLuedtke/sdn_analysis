"""
Name normalization utilities for sanctions matching.

All matchers operate on normalized forms. Normalization here is
intentionally conservative — it preserves meaning while removing
surface variation that shouldn't affect match decisions.
"""

from __future__ import annotations
import re
import unicodedata

# Particles that carry little discriminating signal in name matching.
# Kept as a set for O(1) lookup during tokenization.
_PARTICLES = frozenset({
    "al", "el", "ul", "bin", "bint", "ibn", "abu", "umm",
    "de", "van", "von", "der", "den", "del", "di", "du",
    "le", "la", "les", "los", "las",
})


def normalize(name: str) -> str:
    """
    Return a normalized form of a name string:
    - Lowercase
    - Strip diacritics (NFD decomposition)
    - Remove punctuation, keeping only alphanumeric and spaces
    - Collapse whitespace

    Does NOT remove particles or reorder tokens — those decisions
    belong in the caller.
    """
    if not name:
        return ""
    name = name.lower()
    # Strip diacritics: decompose to NFD, drop combining characters
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Remove punctuation (hyphens, apostrophes, dots, etc.)
    name = re.sub(r"[^\w\s]", " ", name)
    # Collapse whitespace
    return re.sub(r"\s+", " ", name).strip()


def tokenize(name: str) -> list[str]:
    """Normalize and split into tokens."""
    return normalize(name).split()


def tokenize_no_particles(name: str) -> list[str]:
    """Normalize, split, and remove common particles."""
    return [t for t in tokenize(name) if t not in _PARTICLES]
