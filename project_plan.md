# SDN List Analytics Portfolio Project — Build Plan

> **Purpose of this document.** This is a working plan for a four-notebook portfolio project demonstrating sanctions analytics capability in Python. It is intended to be used as the guiding spec for a Claude Code session (or sequence of sessions). The reader/operator is **Ben Luedtke**, an analytics lead with deep Python/SQL/TensorFlow experience and 12+ years of data work, pivoting toward sanctions compliance roles. The audience for the *output* is sanctions hiring managers at US Bank MRM, Sayari, Kharon, Castellum.AI, Guidehouse, Booz Allen threat finance, and similar roles.
>
> **What "done" looks like.** A public GitHub repo containing four progressive Jupyter notebooks, a README that opens with the operational problem before the code. Notebooks 1+2 are publishable independently; the full sequence is ~10–12 weeks at 4–6 hours/week.
>
> **What this is not.** Not a production screening engine. Not a microservice. Not a research paper. Not a vendor pitch. The goal is to demonstrate operational and technical taste, not to invent something new.

---

## Operating principles for the Claude Code session

Read these before starting any notebook.

**Production-mindset, not production code.** Clean module structure, type hints, docstrings, and tests on non-trivial logic. But no Kubernetes, no FastAPI service, no deployment pipeline. Sanctions hiring managers are not impressed by infra theater.

**Explainability over cleverness.** Every model decision must be defensible in a paragraph to a regulator. SR 26-2 and the broader MRM expectation set is the constraint sanctions ML lives under. If a hiring manager asks "why did this score 0.82?", the answer cannot be "the neural net decided." Stick to algorithms whose behavior is interpretable: classical fuzzy matchers, weighted ensembles with documented weights, gradient boosting with SHAP values at most. No fine-tuned transformers in this project.

**Holdout discipline.** No performance number gets reported without a held-out evaluation set. Hiring managers have been burned by vendors who optimized for demo-set accuracy and read portfolio claims with deep skepticism. Build the eval harness early, before the models.

**Operational framing in writeups.** Every notebook opens by stating the operational pain point a sanctions team faces, then introduces the data, then the approach, then the finding. Never open with the algorithm.

**Honest about limits.** If a notebook can only demonstrate something on synthetic data, say so. If a finding has caveats, name them. The goal is credibility with practitioners, not marketing.

**Tooling baseline.**
- Python 3.11+
- `uv` for environment management (faster than poetry, simpler than conda)
- `ruff` for linting and formatting (replaces black + flake8 + isort)
- `pytest` for tests on non-trivial logic
- `lxml` for XML (not stdlib ElementTree — the OFAC schema has nested namespaces ElementTree handles awkwardly)
- `duckdb` or `sqlite` for ad-hoc tabular query
- `pandas` and `polars` are both fine; lean polars for the larger joins
- `rapidfuzz` for fuzzy matching (much faster than fuzzywuzzy, drop-in API)
- `jellyfish` for phonetic encodings (Soundex, Metaphone, NYSIIS, Match Rating)
- `networkx` for graph work in notebook 4 (Neo4j is overkill for portfolio)
- `plotly` or `matplotlib` for visualization — pick one and stick to it
- `jupyter` for the notebooks themselves
- Notebooks render on GitHub natively; do not ship a separate static site

**Repo layout.**
```
sddn_analytics/
├── README.md                    # opens with operational problem, then results, then code
├── pyproject.toml               # uv-managed
├── .ruff.toml
├── data/
│   ├── raw/                     # gitignored; download scripts live here
│   ├── interim/                 # parsed but not analysis-ready
│   └── processed/               # analysis-ready parquet files
├── src/
│   └── sanctions/
│       ├── __init__.py
│       ├── parsers/             # XML parsing: ofac_advanced.py, ofsi.py, eu.py, un.py
│       ├── matching/            # name matching algorithms and ensemble
│       ├── eval/                # precision/recall harness, synthetic query generation
│       └── graph/               # 50% rule traversal
├── notebooks/
│   ├── 01_sdn_eda_and_data_quality.ipynb
│   ├── 02_fuzzy_matching_and_false_positives.ipynb
│   ├── 03_arabic_script_handling.ipynb
│   └── 04_fifty_percent_rule_graph.ipynb
├── tests/
│   └── test_matching.py
└── scripts/
    ├── download_ofac.sh
    ├── download_ofsi.sh
    └── download_opensanctions.py
```

**Reproducibility.** Every notebook should be runnable end-to-end on a fresh clone after `uv sync` and running the download scripts. Pin versions. Cache parsed data to `data/interim/` and `data/processed/` as parquet so notebook re-runs are fast.

**README structure.** Open with one paragraph stating the problem (sanctions screening is the regulatory backbone of cross-border finance, screening engines generate 95-99% false positives, name matching across non-Latin scripts is the largest unsolved technical problem, the 50% rule turns the explicit list into a graph problem). Then results (one screenshot or summary stat per notebook). Then a section per notebook with a one-paragraph summary and a link to the rendered notebook. Setup instructions go at the bottom, not the top.

---

## Data sources

All free and public. Document the URL, the format, the update cadence, and the schema in the parser module docstring.

| Source | URL pattern | Format | Why |
|---|---|---|---|
| OFAC SDN Advanced | `https://www.treasury.gov/ofac/downloads/sdn_advanced.xml` | XML | The structured advanced format with proper entity records, AKAs flagged strong/weak, structured DOB/POB/IDs |
| OFAC Consolidated Advanced | `https://www.treasury.gov/ofac/downloads/cons_advanced.xml` | XML | Non-SDN consolidated lists (SSI, FSE, NS-MBS, CAPTA, PLC) — same schema as SDN advanced |
| OFAC SDN flat CSV | `https://www.treasury.gov/ofac/downloads/sdn.csv` | Pipe-delimited CSV | Use only for comparison — show that you tried it and saw the data loss vs. advanced XML |
| UK OFSI Consolidated | HM Treasury consolidated list page | XML and Excel | Cleanest non-US comparison |
| EU Consolidated Financial Sanctions | EEAS / European Commission FSF | XML | The largest non-US regime by entity count |
| UN Security Council Consolidated | `https://scsanctions.un.org/resources/xml/en/consolidated.xml` | XML | Sparse but globally authoritative |
| OpenSanctions | `https://www.opensanctions.org/` | API + bulk JSON/CSV | Aggregates 200+ lists worldwide; crucially includes some ownership data harvested from OpenCorporates, EU registries, ICIJ leaks. Free tier sufficient. |

Do not commit raw data to the repo. The download scripts in `scripts/` should fetch fresh on demand, and a `data/.gitignore` should exclude everything except `.gitkeep` files.

---

## Notebook 1: SDN Exploratory Data Analysis and Data Quality

**Estimated effort:** 2 weekends, ~12 hours total. **Publishable independently:** yes.

### Operational framing for the writeup

Sanctions screening teams treat the SDN list as ground truth, but the list is itself a real-world dataset with quality variance, schema quirks, and program-specific patterns that drive operational reality. List management is its own discipline: weak AKAs become false positive generators, missing DOBs degrade individual screening precision, missing tax IDs degrade entity screening. Before building a screening engine, you have to know your list. This notebook profiles the OFAC SDN list as if you were inheriting it as a list management lead at a Tier 1 bank.

### Build sequence

1. **Parse the advanced XML cleanly.** Use `lxml`. Build a parser module (`src/sanctions/parsers/ofac_advanced.py`) that produces five normalized tables: `entities` (one row per SDN, with entity type, programs, sanctioning authority, listed date), `akas` (one row per AKA, foreign key to entity, with strong/weak flag and script ID), `addresses`, `documents` (passport, tax ID, IMO, MMSI, etc.), and `features` (DOB, POB, citizenship, nationality, gender). Persist as parquet to `data/processed/`.
2. **Compare against the flat CSV.** Parse `sdn.csv` separately and quantify what is lost. Headline finding will be that strong/weak AKA distinction, structured DOB ranges, and script IDs are all collapsed in the flat format. This is a one-paragraph aside in the writeup but signals that you understand format choices have operational consequences.
3. **Profile by entity type.** How many individuals, entities, vessels, aircraft. Counts and percentages.
4. **Profile by program.** Top 20 programs by entity count. Plot the growth of UKRAINE-EO13662, IRAN, SDGT, SDNTK, CYBER over time using listed dates.
5. **Profile AKA behavior by program.** This is where the interesting findings live. Compute the **mean number of AKAs per entity, broken out by program**, and the **ratio of weak to strong AKAs by program**. Iran-related listings carry far more weak AKAs than Russia listings — this has direct screening implications because weak AKAs are the dominant false positive generator. Visualize as a horizontal bar chart with programs sorted by weak-AKA ratio.
6. **Profile identifier completeness.** What percent of individuals have at least one of: full DOB, year-only DOB, passport number, national ID. What percent of entities have a tax ID, registration number, IMO. Break out by program. The variance here is the operational reality screening teams live with.
7. **List growth over time.** Use the listed-date field to plot cumulative SDN count by month for the last five years. Annotate major events: Feb 2022 invasion (Russia-EO14024 spike), Feb 2025 cartel FTO designations, Oct 2025 Rosneft/Lukoil designations. Optional: archive daily snapshots of the list for two weeks while building this notebook and show the daily delta.
8. **One-paragraph data quality summary.** Cite the Castellum.AI Global Sanctions Index numbers from Part 1 of the briefing as a comparison point — OFAC scores 93% on data integrity, most other lists score below 50%. Position the OFAC profiling work as the cleanest case; downstream multi-list harmonization is harder.

### What this notebook signals

Structured data ingestion from regulatory sources. XML parsing fluency. Understanding that OFAC data is not ground truth but a real-world dataset. Operational awareness of why list management is its own discipline. Visual taste — well-chosen charts, not a wall of seaborn defaults.

### Pitfalls to avoid

Do not start with the flat CSV out of laziness. Do not collapse strong and weak AKAs into a single count. Do not use TF-IDF or embedding methods here — this is profiling, not modeling. Do not include any analysis that depends on a model; that is notebook 2.

---

## Notebook 2: Fuzzy Name Matching and False Positive Analysis

**Estimated effort:** 3 weeks, ~18 hours total. **Publishable independently:** yes.

### Operational framing for the writeup

Sanctions screening generates 95–99% false positive rates in production. Per-alert investigation costs run $30–50 at L1 and $300+ at L3, and a $50B-asset bank generates 5–10M alerts a year. Every algorithmic choice in the matching layer compounds into millions of dollars of operational cost or millions of dollars of regulatory risk if a true hit slips through. This notebook builds a name-matching pipeline against the OFAC SDN list, evaluates it honestly, and shows the precision-recall curve that explains why the entire sanctions screening industry exists.

### Build sequence

1. **Build a clean name corpus.** Use the parsed `akas` table from notebook 1. For each SDN, you have a primary name plus N AKAs. Normalize: lowercase, strip punctuation, collapse whitespace, optionally normalize common particles (de, van, al-, bin, ibn).
2. **Implement three baseline matchers as separate functions.**
   - **Jaro-Winkler** (via `rapidfuzz` or `jellyfish`). Workhorse for individual names; rewards prefix matches.
   - **Token set ratio** (via `rapidfuzz`). Handles word reordering — "Mohammed Ali Hassan" vs. "Hassan, Mohammed Ali."
   - **Phonetic encoding** — Double Metaphone or Beider-Morse (via `jellyfish` or `metaphone`). Catches transliteration variants in Latin script.
3. **Build a candidate generation step (blocking).** Scoring every query against every SDN is wasteful. Implement a simple blocking strategy: index by first letter of first token, by first three characters, or by phonetic key. Document which blocking strategy you chose and the recall implications. Honest blocking degrades recall slightly; quantify the degradation.
4. **Build a weighted ensemble scorer.** Combine the three matchers with documented weights. Do not learn the weights — set them by hand and defend them in a paragraph. The point is interpretability.
5. **Build the eval harness in `src/sanctions/eval/`.** This is the most important code in the project. Construct a synthetic query set with three components:
   - **Positive queries with known SDNs:** take 100 SDN entries, generate 5 variants each (transliteration variation, name reordering, missing patronymic, common misspelling, abbreviated form). 500 positive queries with known correct matches.
   - **Hard negative queries:** the 1,000 most common US baby/family names from SSA data, plus 100 common corporate name fragments (Acme Corp, Global Industries, etc.). Should never match.
   - **Near-miss negatives:** the 200 names most likely to confuse the matcher — common Arabic given names, common Russian patronymics, common Chinese surnames. The ones that *should* score high but resolve to no real SDN.
6. **Run the precision-recall curve.** Sweep the threshold from 0.0 to 1.0 in 0.05 steps. At each threshold compute precision, recall, F1, and false positive rate. Plot precision-recall and ROC. The headline visualization of the entire project.
7. **Compute the operational impact.** At the threshold required to catch 95% of true positives, what is the false positive rate? Multiply by a hypothetical screening volume (1M screenings/month, in line with mid-size bank) and a per-alert cost ($35 average). Show the implied annual investigation budget. This number is always uncomfortably large and is the punchline.
8. **Per-program performance breakdown.** Run the same eval harness segmented by program (Russia, Iran, SDGT, SDNTK). The results will vary substantially — Iran will be hardest. This sets up notebook 3.

### What this notebook signals

String-matching algorithm fluency and judgment about when to use each. Classical ML evaluation discipline (precision-recall, threshold selection, false positive vs. false negative tradeoffs). Operational instinct (the cost calculation). The taste to build the eval harness before the models.

### Pitfalls to avoid

Do not skip the synthetic query generation and report numbers on the SDN list against itself — meaningless. Do not use a single algorithm and call it a comparison. Do not report a single precision/recall number; the curve is the point. Do not learn ensemble weights with a model — interpretability matters more than 2 F1 points. Do not use any deep learning here.

---

## Notebook 3: Arabic Script and Non-Latin Name Handling

**Estimated effort:** 2 weeks, ~12 hours total. **Publishable independently:** yes — this is the most differentiated of the four.

### Operational framing for the writeup

A large share of OFAC SDN designations are individuals and entities with primary names in Arabic, Persian, Russian Cyrillic, Chinese, Korean, or Hebrew script. Screening systems built on Latin-only fuzzy matching handle these by transliterating to Latin first — but there are six common Arabic-to-Latin transliteration standards in active use, and "محمد" can correctly romanize as Mohamed, Mohammed, Muhammad, Mohamad, or Muhammed depending on which standard. A naive Jaro-Winkler treats these as different names and either misses true positives or, with looser thresholds, generates Latin-name false positives at unsustainable rates. This notebook quantifies the problem on the OFAC Iran and counter-terror financing programs and demonstrates a script-aware matcher that improves precision-recall on those programs.

### Build sequence

1. **Extract the multilingual SDN subset.** Use the `scriptId` attribute in the parsed `akas` table to identify entries with Arabic-script primary names or AKAs. Focus on Iran (IRAN, IRAN-HR, IRAN-CON-ARMS-EO), Hezbollah (SDGT entries), and Hamas-related designations. Aim for ~500 entities.
2. **Demonstrate the transliteration variance problem concretely.** Take 20 well-known Arabic-script names from the list. Run them through three transliteration approaches: `unidecode` baseline (fast, lossy), a simple ALA-LC mapping table (you'll write this yourself; it's ~50 lines), and `camel-tools` if you can get it installed (best-in-class Arabic NLP toolkit, occasionally finicky on non-Linux). Show the variants side by side. The visualization here is a table, not a chart.
3. **Build a script-aware canonical form.** Two-stage normalization: (a) if input is Arabic script, normalize Arabic orthography first — strip diacritics (tashkeel), normalize alif variants (ا, أ, إ, آ → ا), normalize ya/alif maqsura (ى → ي), normalize ta marbuta (ة → ه). Then (b) transliterate to a single canonical Latin form using the ALA-LC mapping. Both steps are documented and defensible to a regulator.
4. **Build a matcher that operates on canonical forms.** Take the Latin matcher from notebook 2 and wrap it: query and target are both canonicalized before scoring. For mixed-script queries (a Latin query against an Arabic-script SDN) the canonicalization happens on the SDN side; both end up in the same canonical Latin space.
5. **Run the eval harness from notebook 2 against the Iran subset.** Compare three configurations: Latin-only baseline (notebook 2 matcher with no script awareness), `unidecode` baseline (the lazy production approach), and the script-aware canonical matcher. Report precision-recall at three operational threshold points (high recall, balanced, high precision) for each. The script-aware matcher should win on Iran by a meaningful margin — typically 5–15 points of F1 at the balanced threshold.
6. **Show the failure modes.** Where does the script-aware matcher still struggle? Names with multiple valid Arabic spellings (rare but real), Latin queries with extreme misspellings, and Persian-vs-Arabic ambiguity will all show up. Honest about limits.
7. **Brief discussion of extending to other scripts.** One paragraph each on Cyrillic (BGN/PCGN vs. ISO 9 vs. ALA-LC), Chinese (Pinyin vs. Wade-Giles, simplified vs. traditional), Hebrew (similar transliteration variance), and Korean (Revised Romanization vs. McCune-Reischauer). Do not implement these — describe the same approach scaled to those scripts, and note that production screening engines like Fircosoft and Actimize include language-specific modules that operationalize exactly this pattern.

### What this notebook signals

Multilingual / non-Latin handling, which approximately zero candidates produce. Direct relevance to Iran and counter-terror financing programs, which are perpetual top-five priorities for any sanctions team. Cultural and linguistic specificity — you understand that "transliteration" isn't a single function but a family of standards with operational consequences. If you list Arabic study on the resume (per the earlier conversation), this notebook validates the claim concretely.

### Pitfalls to avoid

Do not claim Arabic fluency in the writeup; you don't need to. The technical demonstration speaks for itself; the resume line ("Arabic — reading script and phonetics, elementary spoken") plus this notebook is more credible than any spoken-fluency claim. Do not fine-tune a multilingual transformer for this — the canonical-form approach is more interpretable, faster, and more defensible to MRM. Do not skip the failure-modes section — practitioners trust honest assessments.

---

## Notebook 4: 50% Rule and Ownership Graph Analysis

**Estimated effort:** 4 weeks, ~24 hours total. **Publishable independently:** yes, but builds on the entity work in notebook 1.

### Operational framing for the writeup

OFAC's 50% rule blocks any entity 50% or more owned, directly, indirectly, or in aggregate, by one or more SDNs — even if not explicitly listed. Industry estimates suggest approximately 95% of effectively sanctioned entities are not on the SDN list. The June 2025 GVA Capital case ($216M, statutory maximum) was OFAC's clearest signal yet that it will pierce trust beneficiary and proxy-control structures. The 50% rule turns sanctions screening from a list lookup into a graph traversal problem, and operationalizing it at scale is what Sayari, Kharon, and Orbis exist to sell. This notebook demonstrates the problem and a basic solution on public ownership data.

### Build sequence

1. **Pull the OpenSanctions dataset.** Use the bulk JSON or the API. Filter to entities and ownership relationships. Focus on a single regime — Russia post-2022 is the richest because of OpenCorporates, EU registry, and ICIJ leak coverage. Iran is also viable. Avoid Venezuela and DPRK; ownership data is sparse.
2. **Build the graph.** Use `networkx`. Nodes: entities (both SDNs and non-SDN companies/individuals). Edges: ownership relationships with weight = ownership percentage. Document the data sources for each edge.
3. **Implement the 50% rule traversal.** This is the substantive code. For each non-SDN node, compute the aggregate SDN ownership across all paths. The traversal:
   - Identify all SDN ancestors of the node (any SDN with a directed ownership path).
   - For each SDN ancestor, compute the product of ownership percentages along each path (a 60% owner of a company that owns 70% of the target contributes 42%).
   - When multiple SDN ancestors exist, sum their contributions (the 50% rule explicitly aggregates across multiple SDNs).
   - Flag any node with aggregate SDN ownership ≥ 50%.
   - Edge cases to handle: cycles (rare in ownership but real), missing ownership percentages (treat as unknown, not zero), and multiple paths from a single SDN (do not double-count).
4. **Quantify the explicit-list iceberg.** Count the explicit SDN nodes in your subset, then count the non-SDN nodes that the 50% rule traversal flags as effectively blocked. Report the ratio. The number is always uncomfortably large.
5. **Pick a real case study for visualization.** Walk through the Kerimov network (the GVA Capital subject, well documented), the Iranian IRGC commercial proxy network, or the Russian sanctioned-oligarch yacht-and-real-estate web. Visualize the graph using `pyvis` or a static `networkx` + `matplotlib` rendering. SDN nodes red, effectively-blocked-by-50%-rule nodes orange, clean nodes gray. Label the path that triggered the rule for one or two non-SDN nodes.
6. **Run a community detection algorithm.** Louvain or Leiden via `networkx` or `python-louvain`. Show that the algorithm tends to surface clusters that map to known sanctioned networks. This is genuinely useful for investigators — it identifies likely sanctions-evasion structures before any individual entity has been designated.
7. **Discuss the operational gap.** OpenSanctions ownership data is far sparser than what Sayari, Kharon, and Orbis maintain commercially. Be explicit about this — the demonstration shows that the 50% rule is solvable as a graph problem on the data you can access; the commercial vendors win on data coverage, not on algorithmic novelty. This honest framing is exactly what a Sayari or Kharon hiring manager wants to see.

### What this notebook signals

Graph data modeling fluency. Understanding that sanctions is a graph problem, which is the entire pitch of Sayari, Kharon, Quantexa, and Palantir. Comfort with the substantive content of recent enforcement actions (GVA Capital). Honest framing about data limitations. The ability to walk through an investigative case study, which is the daily work of an analyst at any of those firms.

### Pitfalls to avoid

Do not use Neo4j here unless you already know it well — the setup overhead is not worth it for a portfolio piece, and `networkx` is sufficient. Do not invent ownership data — every edge must trace to a public source. Do not claim the demonstration matches commercial vendor coverage; it does not, and pretending otherwise destroys credibility. Do not skip the community detection section — it's the part that surprises people.

---

## Cross-cutting concerns

### Testing

Unit tests on:
- The XML parsers (parse a known fixture, assert expected fields).
- The Latin matcher and the script-aware matcher (assert known-good and known-bad pairs).
- The 50% rule traversal (build a synthetic graph with known answers, assert correct flagging including aggregation across multiple SDNs and handling of cycles).

No tests needed on the eval harness itself, but the synthetic query generator should be deterministic given a seed.

### Performance

For the portfolio, performance is not the point — but if a notebook takes more than five minutes to run end-to-end on a laptop, it will not get re-run by reviewers. Cache parsed data as parquet. Use `polars` for the larger joins in notebook 1. Use `rapidfuzz` (C++ backend) rather than `fuzzywuzzy` (pure Python). Block before scoring in notebook 2. Sample the OpenSanctions data to a workable subset in notebook 4 if needed; document the sampling.

### Visualizations

Pick one library and stick to it. Plotly is more interactive but heavier in a notebook; matplotlib renders smaller and reads faster on GitHub. For notebook 1, prefer matplotlib. For notebook 4, `pyvis` for the network visualization (interactive HTML output) is worth the extra dependency. Avoid seaborn defaults; they read as student work. A few well-labeled matplotlib charts with sensible color choices read as professional.

### Documentation

Each notebook ends with a one-paragraph "what this means operationally" closer that ties the technical finding back to the operational pain point opened with. The README and the Substack/Medium posts riff on these closers.

### What to publish on Substack/Medium

One post per notebook, 800–1,500 words. The post is not the notebook — it is the operational story with one or two key visualizations. Title structure: "What [the analysis] taught me about [the operational reality]." Examples:
- "What the OFAC SDN advanced XML taught me about list management"
- "Why sanctions screening generates 99% false positives, and what the precision-recall curve actually looks like"
- "The Arabic transliteration problem in sanctions screening, demonstrated on Iran designations"
- "Why the 50% rule is the entire reason Sayari, Kharon, and Orbis exist"

These titles get clicks from sanctions practitioners on LinkedIn and are the format that will surface to hiring managers.

### Public posture

Make the repo public from the start, not at the end. A repo with five commits is more credible than a repo with one perfect commit, because it shows the work. Use clear commit messages. Do not squash. The progression is part of the signal.

---

## Sequencing for the Claude Code session(s)

A reasonable cadence:

1. **Session 1 (3–4 hours):** Project skeleton. Repo init, `uv` setup, `ruff` config, directory structure, README skeleton, download scripts, parser stub, first 10 lines of notebook 1.
2. **Session 2 (3–4 hours):** Finish notebook 1's parser and EDA. Land the data quality finding. Commit and push.
3. **Session 3 (3–4 hours):** Notebook 2 baseline matchers and synthetic query generator.
4. **Session 4 (3–4 hours):** Notebook 2 ensemble, blocking, eval harness, precision-recall curves, cost calculation. Commit.
5. **Session 5 (2–3 hours):** Notebook 3 Arabic normalization and transliteration. Eval against Iran subset.
6. **Session 6 (2–3 hours):** Notebook 3 polish and writeup. Commit.
7. **Session 7 (3–4 hours):** OpenSanctions ingestion and graph construction.
8. **Session 8 (3–4 hours):** Notebook 4 50% rule traversal, case study visualization, community detection. Commit.
9. **Session 9 (2 hours):** Final README polish, cross-notebook consistency pass, ensure clean run on fresh clone. Tag v1.0.

Each session opens with the operator stating which notebook is being worked on and the specific deliverable for the session. Claude Code should reference this plan rather than reinvent the structure on each turn.

---

## Calibration: what good looks like

After Notebook 1: a hiring manager who clicks through the GitHub repo should think "this person has parsed the OFAC advanced XML for real, not just downloaded it" within 90 seconds.

After Notebook 2: that same hiring manager should think "this person could walk into our screening tuning team and contribute on day one."

After Notebook 3: "this person has thought about the actual hard problem in a way most of our team hasn't."

After Notebook 4: "we should talk to this person about a Solutions Engineer role next week."

If the work clears those bars, it has done its job. The portfolio is not the destination — the conversation it triggers is.
