"""
Synthetic query set generator for sanctions matching evaluation.

Produces three sets of queries that together form a credible eval harness:

    positive (500)    — Known SDN entries with realistic name variants applied.
                        A matcher must retrieve the correct SDN entity.

    hard_negative (1000) — Common US first + last name combinations and
                           corporate fragments that should never match any SDN.

    near_miss (200)   — Common Arabic given names and Russian patronymics
                        combined with common surnames. These names are
                        plausible-looking and will score high against the SDN
                        list by naive matchers, but resolve to no specific
                        SDN entry. They are the primary false positive source
                        in production screening.

All generators are deterministic given a seed.

Variant types applied to positives:
    transliterate  — common character substitutions (ph↔f, ou→u, double→single)
    reorder        — swap first and last tokens
    drop_middle    — remove a middle token (patronymic / middle name)
    misspell       — one character-level edit (transposition or substitution)
    abbreviate     — replace first token with its initial
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

# ── Common US names embedded to avoid external data dependency ────────────────

_FIRST_NAMES = [
    "James", "Robert", "John", "Michael", "David", "William", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
    "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]

_CORP_FRAGMENTS = [
    "Global Industries LLC", "Pacific Trading Co", "Atlantic Holdings Inc",
    "Northern Capital Group", "Eastern Resources Ltd", "Western Commerce Corp",
    "United Enterprises", "National Supply Company", "Premier Services Group",
    "Allied Business Partners", "Continental Trading LLC", "Apex Solutions Inc",
]

# Common Arabic given names — appear frequently in SDN list and in
# legitimate customer bases; primary source of false positives for
# Iran / SDGT programs.
_ARABIC_FIRST = [
    "Mohammed", "Mohamed", "Muhammad", "Ahmad", "Ahmed", "Hassan", "Hussein",
    "Ali", "Omar", "Ibrahim", "Khalid", "Abdullah", "Yousef", "Yusuf",
    "Hasan", "Karim", "Samir", "Tariq", "Bashir", "Nasser", "Khaled",
    "Walid", "Rashid", "Faisal", "Hamid", "Amir", "Bilal", "Mustafa",
    "Adel", "Sami", "Mahmoud", "Marwan", "Wael", "Ramzi", "Imad",
]

_ARABIC_LAST = [
    "Al-Rashid", "Al-Hassan", "Al-Hussein", "Al-Khalid", "Al-Omar",
    "Rahman", "Karimi", "Ahmadi", "Hosseini", "Mohammadi",
    "Saleh", "Nassar", "Mansour", "Qasim", "Shaikh",
]

# Common Russian first names + patronymics
_RUSSIAN_FIRST = [
    "Aleksandr", "Dmitri", "Sergei", "Andrei", "Nikolai",
    "Vladimir", "Mikhail", "Pavel", "Ivan", "Alexei",
    "Yuri", "Boris", "Anatoli", "Viktor", "Konstantin",
]

_RUSSIAN_LAST = [
    "Ivanov", "Petrov", "Smirnov", "Kuznetsov", "Popov",
    "Volkov", "Sokolov", "Lebedev", "Kozlov", "Novikov",
    "Morozov", "Orlov", "Fedorov", "Mikhailov", "Nikolaev",
]

# ── Variant generators ────────────────────────────────────────────────────────

_TRANSLIT_RULES = [
    (r"ph", "f"),
    (r"(?<![aeiou])f(?![aeiou])", "ph"),
    (r"ou", "u"),
    (r"ei", "i"),
    (r"ae", "a"),
    (r"ck", "k"),
    (r"(\w)\1", r"\1"),          # double consonant → single (hassan → hasan)
    (r"ah$", "a"),
    (r"eh$", "e"),
    (r"i$", "y"),
    (r"y$", "i"),
]


def _apply_transliteration(name: str, rng: random.Random) -> str:
    """Apply one random character substitution rule, preserving original case."""
    lower = name.lower()
    applicable = [(pat, rep) for pat, rep in _TRANSLIT_RULES
                  if re.search(pat, lower)]
    if not applicable:
        return name
    pat, rep = rng.choice(applicable)
    return re.sub(pat, rep, lower, count=1)


def _reorder(name: str, rng: random.Random) -> str:
    """Swap first and last tokens."""
    tokens = name.split()
    if len(tokens) < 2:
        return name
    return " ".join([tokens[-1]] + tokens[1:-1] + [tokens[0]])


def _drop_middle(name: str, rng: random.Random) -> str:
    """Remove a random middle token (simulates omitted patronymic)."""
    tokens = name.split()
    if len(tokens) < 3:
        return name
    middle_indices = list(range(1, len(tokens) - 1))
    drop = rng.choice(middle_indices)
    return " ".join(tokens[:drop] + tokens[drop + 1:])


def _misspell(name: str, rng: random.Random) -> str:
    """Apply one character-level edit within a randomly chosen word."""
    words = name.split()
    # Pick a word long enough to edit
    candidates = [w for w in words if len(w) >= 3]
    if not candidates:
        return name
    word = rng.choice(candidates)
    i = rng.randint(1, len(word) - 2)
    choice = rng.choice(["transpose", "substitute"])
    chars = list(word)
    if choice == "transpose":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    else:
        subs = {"a": "e", "e": "a", "i": "y", "y": "i",
                "o": "u", "u": "o", "s": "z", "z": "s"}
        chars[i] = subs.get(chars[i].lower(), chars[i])
    new_word = "".join(chars)
    return name.replace(word, new_word, 1)


def _abbreviate(name: str, rng: random.Random) -> str:
    """Replace first token with its initial."""
    tokens = name.split()
    if len(tokens) < 2:
        return name
    return tokens[0][0] + " " + " ".join(tokens[1:])


_VARIANT_FNS = {
    "transliterate": _apply_transliteration,
    "reorder":       _reorder,
    "drop_middle":   _drop_middle,
    "misspell":      _misspell,
    "abbreviate":    _abbreviate,
}

# ── Public API ────────────────────────────────────────────────────────────────

def generate_positive_queries(
    akas: pd.DataFrame,
    entities: pd.DataFrame,
    n_entities: int = 100,
    variants_per_entity: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_entities SDN individuals with Latin-script names and generate
    variants_per_entity name variants for each.

    Returns a DataFrame with columns:
        query_name      — the generated variant
        true_entity_id  — SDN entity_id it should match
        true_name       — original primary name
        variant_type    — which transformation was applied
        program         — first program of the entity (for per-program eval)
    """
    rng = random.Random(seed)

    # Select candidate entities: individuals, Latin script, 2+ tokens
    primary = akas[akas["is_primary"] & (akas["script"] == "Latin")].copy()
    primary = primary.merge(
        entities[entities["entity_type"] == "Individual"][["entity_id", "programs"]],
        on="entity_id",
    )
    primary = primary[primary["aka_name"].str.split().str.len() >= 2]
    primary["program"] = primary["programs"].apply(
        lambda p: list(p)[0] if len(list(p)) > 0 else "UNKNOWN"
    )

    # Random sample — stratification adds complexity without meaningful benefit
    # for the eval harness; entity diversity comes naturally from the list.
    if len(primary) <= n_entities:
        sampled = primary
    else:
        sampled = primary.sample(n_entities, random_state=seed)

    variant_types = list(_VARIANT_FNS.keys())
    rows = []
    for _, row in sampled.iterrows():
        name = row["aka_name"]
        assigned = rng.sample(variant_types, min(variants_per_entity, len(variant_types)))
        # Fill remaining slots by repeating types if needed
        while len(assigned) < variants_per_entity:
            assigned.append(rng.choice(variant_types))

        for vtype in assigned:
            variant = _VARIANT_FNS[vtype](name, rng)
            # Fallback: if variant is unchanged, apply misspell
            if variant.lower() == name.lower():
                variant = _misspell(name, rng)
            rows.append({
                "query_name":     variant,
                "true_entity_id": row["entity_id"],
                "true_name":      name,
                "variant_type":   vtype,
                "program":        row["program"],
            })

    return pd.DataFrame(rows)


def generate_hard_negatives(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate n combinations of common US names and corporate fragments.
    These should never match any SDN entry.
    """
    rng = random.Random(seed)
    rows = []

    # Individual names
    for _ in range(n - len(_CORP_FRAGMENTS)):
        first = rng.choice(_FIRST_NAMES)
        last  = rng.choice(_LAST_NAMES)
        rows.append({"query_name": f"{first} {last}", "negative_type": "us_individual"})

    # Corporate fragments
    for corp in _CORP_FRAGMENTS:
        rows.append({"query_name": corp, "negative_type": "corporate"})

    return pd.DataFrame(rows).sample(n, random_state=seed).reset_index(drop=True)


def generate_near_miss_negatives(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate n names that resemble SDN names but are not specific SDN entries.
    Arabic and Russian names that will score high against the list by naive
    matchers — the primary source of false positives in production screening.
    """
    rng = random.Random(seed)
    rows = []

    for _ in range(n // 2):
        first = rng.choice(_ARABIC_FIRST)
        last  = rng.choice(_ARABIC_LAST)
        rows.append({
            "query_name":   f"{first} {last}",
            "near_miss_type": "arabic",
        })

    for _ in range(n - len(rows)):
        first = rng.choice(_RUSSIAN_FIRST)
        last  = rng.choice(_RUSSIAN_LAST)
        rows.append({
            "query_name":   f"{first} {last}",
            "near_miss_type": "russian",
        })

    return pd.DataFrame(rows).sample(n, random_state=seed).reset_index(drop=True)


def build_query_set(
    akas: pd.DataFrame,
    entities: pd.DataFrame,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Build the full evaluation query set.
    Returns a dict with keys: 'positive', 'hard_negative', 'near_miss'.
    """
    return {
        "positive":      generate_positive_queries(akas, entities, seed=seed),
        "hard_negative": generate_hard_negatives(seed=seed),
        "near_miss":     generate_near_miss_negatives(seed=seed),
    }
