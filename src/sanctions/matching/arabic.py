"""
Arabic name handling for sanctions screening.

Two distinct problems addressed here:

1. ARABIC-SCRIPT INPUT → LATIN CORPUS
   When a customer name arrives in Arabic script and the SDN corpus
   contains only Latin transliterations (as OFAC's Advanced XML does):
   normalize Arabic orthography first, then transliterate to a canonical
   Latin form using a simplified ALA-LC table.

   Functions: is_arabic_script, normalize_arabic_orthography,
              transliterate_ala_lc, arabic_to_canonical_latin

2. LATIN-SCRIPT ARABIC-ORIGIN NAMES → CANONICAL LATIN
   When both query and corpus are already in Latin script but use
   different romanization conventions (the dominant case in OFAC data):
   normalize common transliteration variants to a single canonical form.

   Functions: normalize_arabic_latin_variants

Operational context:
   OFAC's SDN Advanced XML stores all names in Latin transliteration.
   The transliteration variance problem — "SOLEYMANI" vs "SALIMANI",
   "HOSSEIN" vs "HOSEYN", "AL-ANBYA" vs "OL AMBIA" — exists entirely
   in the Latin layer. Both normalization paths ultimately produce a
   canonical Latin form that the existing matchers (NB2) can then score.

References:
   ALA-LC Romanization Tables: Library of Congress, 2012 edition.
   Arabic table: https://www.loc.gov/catdir/cpso/romanization/arabic.pdf
"""

from __future__ import annotations
import re
import unicodedata

# ── Arabic-script detection ────────────────────────────────────────────────────

_ARABIC_RANGES = [
    (0x0600, 0x06FF),   # Arabic block
    (0x0750, 0x077F),   # Arabic Supplement
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
]


def is_arabic_script(text: str) -> bool:
    """Return True if text contains Arabic-script characters."""
    for char in text:
        cp = ord(char)
        if any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES):
            return True
    return False


# ── Arabic orthographic normalization ─────────────────────────────────────────
# Applied before transliteration to reduce orthographic variants in Arabic script.

_ARABIC_DIACRITICS = frozenset(
    "ًٌٍَُِّْٓ"  # harakat
    "ٰ"      # superscript alif
    "ـ"      # tatweel / kashida
)

# Alif variants → bare alif ا (U+0627)
_ALIF_MAP = {
    "أ": "ا",   # أ → ا
    "إ": "ا",   # إ → ا
    "آ": "ا",   # آ → ا
    "ٱ": "ا",   # ٱ → ا
}
# Alif maqsura → ya ي (U+064A)
_ALIF_MAQSURA = {"ى": "ي"}   # ى → ي
# Ta marbuta → ha ه (U+0647)
_TA_MARBUTA    = {"ة": "ه"}  # ة → ه
# Hamza-on-waw / hamza-on-ya → simpler forms
_HAMZA_MAP = {
    "ؤ": "و",   # ؤ → و
    "ئ": "ي",   # ئ → ي
}


def normalize_arabic_orthography(text: str) -> str:
    """
    Normalize Arabic-script text to a reduced orthographic form:
      1. Remove diacritics (tashkeel) and tatweel
      2. Alif variants → ا
      3. Alif maqsura → ي
      4. Ta marbuta → ه
      5. Hamza variants simplified
    This reduces orthographic noise before transliteration.
    """
    result = []
    for ch in text:
        if ch in _ARABIC_DIACRITICS:
            continue
        ch = _ALIF_MAP.get(ch, ch)
        ch = _ALIF_MAQSURA.get(ch, ch)
        ch = _TA_MARBUTA.get(ch, ch)
        ch = _HAMZA_MAP.get(ch, ch)
        result.append(ch)
    return "".join(result)


# ── ALA-LC transliteration table (simplified) ─────────────────────────────────
# Context-sensitive rules (long vowels, sun/moon letter assimilation) are not
# implemented. Sufficient for name matching; not sufficient for scholarly use.

_ALA_LC: dict[str, str] = {
    # Consonants
    "ب": "b",   # ب
    "ت": "t",   # ت
    "ث": "th",  # ث
    "ج": "j",   # ج
    "ح": "h",   # ح  (ḥ simplified to h)
    "خ": "kh",  # خ
    "د": "d",   # د
    "ذ": "dh",  # ذ
    "ر": "r",   # ر
    "ز": "z",   # ز
    "س": "s",   # س
    "ش": "sh",  # ش
    "ص": "s",   # ص  (ṣ simplified to s)
    "ض": "d",   # ض  (ḍ simplified to d)
    "ط": "t",   # ط  (ṭ simplified to t)
    "ظ": "z",   # ظ  (ẓ simplified to z)
    "ع": "",    # ع  (ayn — often dropped in practical transliteration)
    "غ": "gh",  # غ
    "ف": "f",   # ف
    "ق": "q",   # ق
    "ك": "k",   # ك
    "ل": "l",   # ل
    "م": "m",   # م
    "ن": "n",   # ن
    "ه": "h",   # ه
    "و": "w",   # و  (consonant; long vowel ū not distinguished without diacritics)
    "ي": "y",   # ي  (consonant; long vowel ī not distinguished without diacritics)
    "ء": "",    # ء  (hamza — often dropped)
    # Alif (bare, after normalize_arabic_orthography)
    "ا": "a",
    # Numerals
    **{chr(0x0660 + i): str(i) for i in range(10)},
}


def transliterate_ala_lc(text: str) -> str:
    """
    Transliterate Arabic-script text to Latin using a simplified ALA-LC table.
    Apply normalize_arabic_orthography first for best results.
    """
    result = []
    for char in text:
        if char == " ":
            result.append(" ")
        else:
            result.append(_ALA_LC.get(char, char))
    return re.sub(r"\s+", " ", "".join(result)).strip()


def arabic_to_canonical_latin(text: str) -> str:
    """
    Full pipeline for Arabic-script input:
      normalize orthography → transliterate ALA-LC → lowercase
    """
    normalized = normalize_arabic_orthography(text)
    transliterated = transliterate_ala_lc(normalized)
    return transliterated.lower().strip()


# ── Canonical normalization for Arabic-origin names in Latin script ────────────
# Applied when both query and corpus are already in Latin transliteration but
# may use different romanization conventions (the OFAC case).

# Well-known Arabic given name variants → canonical form
_NAME_CANONICAL: dict[str, str] = {
    # Mohammed
    "mohammed": "muhammad", "mohamed": "muhammad", "mohamad": "muhammad",
    "muhammed": "muhammad", "mohammad": "muhammad", "mehmet": "muhammad",
    "mahomet": "muhammad",
    # Hussein
    "hussein": "husayn", "husain": "husayn", "hossein": "husayn",
    "hoseyn": "husayn", "hussain": "husayn", "hüseyin": "husayn",
    # Hassan
    "hassan": "hasan", "hassen": "hasan",
    # Ahmad
    "ahmed": "ahmad", "ahmet": "ahmad",
    # Qasim
    "qasem": "qasim", "ghasem": "qasim", "kasem": "qasim",
    "kassem": "qasim", "ghasim": "qasim", "qassem": "qasim",
    # Ali
    "aly": "ali",
    # Omar
    "omar": "umar",
    # Yusuf
    "yousef": "yusuf", "yousuf": "yusuf", "youssuf": "yusuf",
    # Ibrahim
    "ebrahim": "ibrahim",
    # US/UK spelling variants common in org names
    "defense": "defence",
    "organization": "organisation",
    "organizations": "organisations",
}

# Suffix normalization (applied per token)
_SUFFIX_RULES: list[tuple[str, str]] = [
    (r"oo$",   "u"),    # oo → u  (Malouf → Maluf)
    (r"ou$",   "u"),    # ou → u  (Mousa → Musa)
    (r"ee$",   "i"),    # ee → i
    (r"ah$",   "a"),    # -ah → -a  (Fatimah → Fatima)
    (r"eh$",   "a"),    # -eh → -a  (Fatimeh → Fatima)
    (r"ain$",  "ayn"),  # -ain → -ayn
    (r"ein$",  "ayn"),  # -ein → -ayn
    (r"eyn$",  "ayn"),  # -eyn → -ayn
]

# Infix vowel rules (applied per token)
_INFIX_RULES: list[tuple[str, str]] = [
    (r"ey(?=[^aeiou])", "i"),   # ey → i interior (Hoseyn → Hosin)
    (r"ou(?=[^aeiou])", "u"),   # ou → u interior (Mousa → Musa)
    (r"oo(?=[^aeiou])", "u"),   # oo → u interior
]

# Article prefix — normalized per token (handles "ol ambia" → "al ambia")
_ARTICLE_TOKENS = frozenset({"al", "el", "ul", "ol"})

# Consonant pattern normalization (within tokens)
_CONSONANT_RULES: list[tuple[str, str]] = [
    (r"([bcdfghjklmnpqrstvwxyz])\1", r"\1"),  # double consonant → single
]


def normalize_arabic_latin_variants(name: str) -> str:
    """
    Normalize Arabic-origin names in Latin script to a canonical form.

    Reduces common romanization variants so that "SOLEYMANI / SALIMANI",
    "HOSSEIN / HOSEYN", "KHATAM AL-ANBYA / KHATAM OL AMBIA" and similar
    pairs map to the same canonical string, improving recall for the
    standard Latin matchers from Notebook 2.

    Does NOT require Arabic-script input — operates entirely in Latin.
    """
    # Step 1: lowercase and strip diacritics
    text = name.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove punctuation except hyphens (preserve al-)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 2: per-token normalization
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        # 2a: known name lookup
        token = _NAME_CANONICAL.get(token, token)
        # 2b: normalize standalone article tokens (ol/el/ul → al)
        if token in _ARTICLE_TOKENS:
            token = "al"
        # 2c: strip leading "al-" / "el-" / "ol-" prefix on compound tokens
        token = re.sub(r"^(el|ul|ol)-", "al-", token)
        # 2d: infix vowel rules
        for pat, rep in _INFIX_RULES:
            token = re.sub(pat, rep, token)
        # 2e: suffix rules
        for pat, rep in _SUFFIX_RULES:
            token = re.sub(pat, rep, token)
        # 2f: double consonant → single
        for pat, rep in _CONSONANT_RULES:
            token = re.sub(pat, rep, token)
        new_tokens.append(token)
    text = " ".join(new_tokens)

    return re.sub(r"\s+", " ", text).strip()


def canonical_form(name: str) -> str:
    """
    Unified canonical form: detect script and apply appropriate pipeline.
    - Arabic script → normalize orthography → ALA-LC transliteration
    - Latin script  → normalize Arabic-Latin variants
    Both paths produce a canonical Latin string for scoring.
    """
    if is_arabic_script(name):
        return arabic_to_canonical_latin(name)
    return normalize_arabic_latin_variants(name)
