# Sanctions Analytics

Sanctions screening is the regulatory backbone of cross-border finance. Every bank, broker-dealer, and payments processor must screen customers and transactions against government watchlists — primarily the OFAC SDN list — before executing. In production, these engines generate **95–99% false positive rates**: for every genuine hit, analysts investigate 100–200 names that aren't on any list. At $30–50 per alert at L1 and $300+ at L3, a mid-size bank's screening program costs tens of millions of dollars a year in investigation labor alone.

The technical problems driving that false positive rate are specific and solvable. Name matching across non-Latin scripts — Arabic, Cyrillic, Chinese, Hebrew — is the hardest: there are six common Arabic-to-Latin transliteration standards in active use, and a screening engine that doesn't account for them either misses true hits or drowns analysts in noise. The OFAC 50% rule compounds the problem further: any entity 50%+ owned by a sanctioned party is effectively blocked even if it isn't on the list, which means the explicit SDN list is the tip of an ownership graph iceberg. Industry estimates put ~95% of effectively sanctioned entities off-list.

This project demonstrates those problems and their solutions across four progressive notebooks, built against live OFAC data using the Advanced XML format.

---

## Results

| Notebook | Finding |
|---|---|
| [01 — SDN EDA and Data Quality](#notebook-1-sdn-eda-and-data-quality) | 17.8% of SDN aliases are flagged low-quality; RUSSIA-EO14024 added 6,393 entities since Feb 2022 — more than IRAN and SDGT combined |
| [02 — Fuzzy Name Matching](#notebook-2-fuzzy-name-matching-and-false-positive-analysis) | *(coming soon)* |
| [03 — Arabic Script Handling](#notebook-3-arabic-script-and-non-latin-name-handling) | *(coming soon)* |
| [04 — 50% Rule Graph Analysis](#notebook-4-50-rule-and-ownership-graph-analysis) | *(coming soon)* |

---

## Notebook 1: SDN EDA and Data Quality

The SDN list is a real-world dataset with program-specific quality variance that determines what any screening engine can and cannot do. This notebook profiles the list as an incoming list management lead would: entity composition, program growth over time, alias structure by program, and identifier completeness by program.

**Key findings:** 17.8% of all SDN aliases (4,380 of 24,644) are flagged low-quality — transliteration variants and phonetic spellings that must be included to catch evasion but match legitimate names at higher rates than strong aliases. Identifier coverage is better than commonly assumed: 99% of SDN individuals have at least a birthdate or government-issued ID on record. The 1% without any structured identifier are screened almost entirely on name — which is where false positives live. The flat CSV (`sdn.csv`) used by most production implementations loses the strong/weak alias distinction, script identification, and structured DOB ranges entirely.

→ [`notebooks/01_sdn_eda_and_data_quality.ipynb`](notebooks/01_sdn_eda_and_data_quality.ipynb)

---

## Notebook 2: Fuzzy Name Matching and False Positive Analysis

*(coming soon)*

Builds a name-matching pipeline against the SDN list using Jaro-Winkler, token set ratio, and phonetic encoding. Constructs a synthetic query set of 500 positive queries (known SDN variants) and 1,200 negative queries (common names, near-miss negatives). Sweeps the precision-recall curve from threshold 0 to 1.0. The headline visualization shows why the sanctions screening industry exists: the false positive rate at 95% recall implies an annual investigation budget no one wants to calculate out loud.

→ [`notebooks/02_fuzzy_matching_and_false_positives.ipynb`](notebooks/02_fuzzy_matching_and_false_positives.ipynb) *(pending)*

---

## Notebook 3: Arabic Script and Non-Latin Name Handling

*(coming soon)*

Extracts the multilingual SDN subset (Iran, Hezbollah-related SDGT, Hamas-related designations — ~500 entities). Demonstrates that "محمد" correctly romanizes as Mohamed, Mohammed, Muhammad, Mohamad, or Muhammed depending on transliteration standard, and that naive Jaro-Winkler treats these as different names. Builds a script-aware canonical-form matcher using Arabic orthographic normalization + ALA-LC transliteration. Runs the eval harness from Notebook 2 against the Iran subset; the script-aware matcher improves F1 at the balanced threshold.

→ [`notebooks/03_arabic_script_handling.ipynb`](notebooks/03_arabic_script_handling.ipynb) *(pending)*

---

## Notebook 4: 50% Rule and Ownership Graph Analysis

*(coming soon)*

Demonstrates that sanctions screening is a graph traversal problem, not a list lookup. Pulls OpenSanctions ownership data for Russia post-2022, builds a directed ownership graph in NetworkX, and implements the OFAC 50% rule traversal: for each non-SDN node, compute aggregate SDN ownership across all paths, flag any node ≥ 50%. Quantifies the explicit-list iceberg. Walks through a real case study (the Kerimov network). Runs Louvain community detection; the clusters map to known sanctioned networks. Discusses frankly where commercial vendors (Sayari, Kharon, Orbis) win on data coverage rather than algorithmic novelty.

→ [`notebooks/04_fifty_percent_rule_graph.ipynb`](notebooks/04_fifty_percent_rule_graph.ipynb) *(pending)*

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Install uv (if not already installed)
brew install uv                 # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url>
cd sanctions-analytics
uv sync

# Download the OFAC data (~15 MB)
bash scripts/download_ofac.sh

# Launch Jupyter
uv run --env-file .env jupyter notebook
```

The download script fetches `SDN_ADVANCED.XML` and `CONS_ADVANCED.XML` from the OFAC Sanctions List Service. These files are gitignored; run the script on a fresh clone before opening any notebook. Parsed parquet files are written to `data/processed/` on first notebook run and cached thereafter.

**Note on Python 3.14:** A known issue with setuptools editable installs on Python 3.14 requires the `--env-file .env` flag for `uv run`. This sets `PYTHONPATH=src` and will be unnecessary once the upstream bug is resolved. Python 3.11–3.13 are unaffected.

---

## A note on AI-assisted development

This project was built with Claude (Anthropic) as a coding collaborator. The analytical decisions, research, and interpretation of findings are mine — Claude assisted with implementation: debugging the OFAC Advanced XML namespace structure, scaffolding the parser and project layout, and accelerating chart-building iteration.
