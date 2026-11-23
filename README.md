# Productivity metrics and developer mental health — replication package

Replication package for **"Individual or Team-Level Metrics? Insights from Developer
Discourse and a Practitioner Survey"**: the analysis code, the query keywords, the
survey instrument and the paper's figures. The pseudonymized dataset is on Zenodo
(link below).

A mixed-methods study with two independent evidence sources:

1. **Reddit mining (primary, observational).** 278,131 messages (25,120 posts +
   253,011 comments) from five developer subreddits, 2021–2024, built from the
   public Pushshift/Academic Torrents archives. Scored with VADER, re-scored with
   RoBERTa, and analysed with a thread-clustered logistic regression.
2. **Practitioner survey (N = 24).** A structured instrument on which metrics
   developers use, trust, and wish existed (`SURVEY_INSTRUMENT.md`).

The central hypothesis is that **individual-level metrics attract more negative
developer discourse than team-level metrics** (the DORA delivery family).

**Data:** the pseudonymized data release is on Zenodo,
[doi:10.5281/zenodo.24450558](https://doi.org/10.5281/zenodo.24450558).

---

```bash
make install     # analysis dependencies (enough for everything below)
make smoke       # full end-to-end run on SYNTHETIC data — no downloads needed
make verify      # byte-compile + lint every script
```

`make install-full` adds the heavy dependencies needed only to rebuild the corpus,
the LLM catalog and the RoBERTa labels from the raw dumps (`requirements-full.txt`:
llama-cpp-python, torch, transformers, libtorrent, …; two of them need native builds).

---

## Reproducing the results

There are two routes. Both end in the same two commands (steps 5 and 6 below).

**A. From the data release** (pseudonymized working data, on Zenodo:
[doi:10.5281/zenodo.24450558](https://doi.org/10.5281/zenodo.24450558)). Download all
of its files into `release/`, then:

```bash
python scripts/analyze_corpus.py --corpus release/corpus_pseudonymized.csv.gz \
    --roberta release/roberta_sentiment.jsonl \
    --curated-catalog release/metricas_catalogadas_filtradas.csv \
    --full-catalog release/metricas_catalogadas.csv \
    --surveillance-review release/surveillance_manual_review.csv
python scripts/analyze_survey.py --csv release/survey_results.csv
```

**B. From scratch** (raw dumps → corpus → labels → results): steps 1–6 below, with
`make install-full`. Steps 3 and 4 are slow without a GPU, and step 3 ends in a manual
curation pass (see *Manual inputs*), so a from-scratch run reproduces the catalog
extraction but not the curated catalog itself.

---

## Pipeline

```
                    Academic Torrents dumps
                    (5 x _submissions.zst, _comments.zst)
                              |
                 [1] download_offline.py     ~206 MB / ~1.5 GB
                              |
                 [2] build_corpus.py  ------>  data/<timestamp>_corpus.csv
                     filters 2021-2024 by the query keywords,
                     tags post/comment, SHA-256 hashes authors,
                     drops permalinks, truncates bodies to 500 chars
                              |
        +---------------------+----------------------+
        |                                            |
 [3] llm_catalog.py  (local Gemma 3 4B)     [4] score_roberta.py
     consolidate_catalog.py                        data/roberta_sentiment.jsonl
     -> data/llm_labels.jsonl
     -> data/metricas_catalogadas.csv
     + manual curation
     -> data/metricas_catalogadas_filtradas.csv
        |                                            |
        +---------------------+----------------------+
                              |
                 [5] analyze_corpus.py   <-- every Reddit number in the paper
                     writes docs/figures/*.pdf
                     writes data/metricas_analise_sentimento.csv
                              |
                 [6] analyze_survey.py   <-- every survey number in the paper
                     data/survey_results.csv -> survey figures
```

Steps 3, 4 and 6 are independent of each other. Step 5 produces every Reddit
number in the paper (Section IV-A and IV-B, Tables III–VI, Figs. 1–2) and the
supplementary corpus figures; step 6 produces every survey number (Section IV-C and
IV-D, Table VII) and the supplementary survey figures. Two inputs of step 5 are manual
(see below).

### Commands

```bash
# 1. fetch the dumps (optional — only needed to rebuild the corpus from scratch)
python scripts/download_offline.py --comments

# 2. build the corpus
python scripts/build_corpus.py                 # -> data/<timestamp>_corpus.csv

# 3. metric catalog: open LLM extraction, then canonicalisation.
#    Needs llama-cpp-python and the Gemma 3 4B instruct model, 4-bit quantized
#    (gemma-3-4b-it-Q4_K_M.gguf, e.g. from huggingface.co/unsloth/gemma-3-4b-it-GGUF),
#    placed in models/ (gitignored). Runs on CPU with --n-gpu-layers 0, but slowly.
python scripts/llm_catalog.py --model models/gemma-3-4b-it-Q4_K_M.gguf
python scripts/consolidate_catalog.py

# 4. second sentiment model, for the robustness table
python scripts/score_roberta.py                # resumable; ~34k docs

# 5. all corpus results and figures
python scripts/analyze_corpus.py

# 6. survey results and figures
python scripts/analyze_survey.py
```

`make download corpus catalog roberta figures survey` runs steps 1–6 in order
(`make catalog MODEL=path/to/model.gguf` to use another model path).
`make all` runs 5 and 6.

The input and output paths of scripts 2, 4, 5 and 6 are CLI flags (`--archive-dir`,
`--out`, `--corpus`, `--roberta`, `--csv`, …), which is what lets the smoke test run
safely against synthetic data. Run any script with `-h` for the full list.

### Manual inputs

Two inputs of step 5 are the product of human judgment, not of a script:

- **Curated catalog** (`data/metricas_catalogadas_filtradas.csv`). A manual pass
  over the consolidated LLM catalog keeps the study-relevant items and labels each as
  productivity, mental health, or both. It is the catalog of record for Fig. 1.
  The curation also removed several frequently extracted items, among them
  most of the DORA family, so `analyze_corpus.py` compares systemic and
  individual-output metrics on the **un-curated** catalog and prints which systemic
  items the curation removed.
- **Employer-monitoring review** (`data/surveillance_manual_review.csv`). A broad
  keyword rule (`track|monitor|surveillance|spy`) matches 1,119 posts, most of them in
  unrelated senses (infrastructure monitoring, "on track", issue tracking). Candidate
  posts found by a monitoring vocabulary and by an employer-context search (263 in all)
  were reviewed one by one;
  `employer_monitoring` records the decision (66 posts). `analyze_corpus.py` reports
  the reviewed count and draws the supplementary employer-monitoring figure from it;
  without the file it falls back to the broad, unreviewed count and labels the figure accordingly.

Both files ship with the data release.

---

## Data provenance

| Path | Origin | In git? |
|---|---|---|
| `data/<timestamp>_corpus.csv` | Derived from the public Pushshift dumps. **Not redistributed.** | no (`data/` is gitignored) |
| `data/reddit/subreddits24/*.zst` | Academic Torrents, dump `1614740a…` | no |
| `data/survey_results.csv` | Manual Google Forms export | no |
| `data/llm_labels.jsonl` | Per-post output of step 3 (LLM extraction) | no |
| `data/metricas_catalogadas.csv` | Produced by step 3 (consolidation) | no |
| `data/metricas_catalogadas_filtradas.csv` | Manual curation of the above | no |
| `data/surveillance_manual_review.csv` | Manual review of employer-monitoring posts | no |
| `data/roberta_sentiment.jsonl` | Produced by step 4 | no |
| `data/metricas_analise_sentimento.csv` | Per-metric table, written by step 5 | no |
| `data/mock/` | Synthetic, generated by `generate_mock_data.py` | no |
| `docs/figures/*.pdf` | Produced by step 5 / 6 | no |
| `results/*.pdf` | Figs. 1–2 as published, and the supplementary figures (see *Figures*) | **yes** |

The raw dumps are independently available via Academic Torrents, so the corpus
is reproducible in principle. Author identifiers are SHA-256 hashed and
permalinks are dropped **before** analysis, so the working dataset is
publishable; only aggregate statistics and non-identifying excerpts appear in
the paper.

`data/` and `docs/` are gitignored, which means **a fresh clone has no data.**
Run `make smoke` to check the code, and use the data release (route A) or the full
pipeline (route B) to reproduce the numbers. `scripts/pseudonymize.py` builds the data
release from the working data.

---

## Repository layout

```
scripts/
  analyze_corpus.py      all corpus statistics + figures   <- source of truth
  analyze_survey.py      all survey statistics + figures
  build_corpus.py        dumps -> data/<timestamp>_corpus.csv
  download_offline.py    fetch the five subreddit dumps
  llm_catalog.py         LLM open extraction over posts
  consolidate_catalog.py canonicalise + aggregate extractions
  score_roberta.py       second sentiment model
  pseudonymize.py        build the data release from the working data
  generate_mock_data.py  synthetic corpus/survey for smoke tests
results/                 the paper's figures, as published (see Figures below)
QUERY_KEYWORDS.md        the two query-keyword lists, their matching rules, and the
                         terms of the three metric categories
query_keywords.csv       the query-keyword lists, machine-readable
SURVEY_INSTRUMENT.md     the practitioner survey (Table II), with the original
                         Portuguese wording
Makefile                 pipeline driver (`make help`)
requirements.txt         analysis dependencies
requirements-full.txt    + corpus / catalog / RoBERTa / download dependencies
```

Running step 5 or 6 writes fresh figures to `docs/figures/`; `results/` holds the
copies used in the paper and is not overwritten by any script.

### Figures

| File | In the paper |
|---|---|
| `results/metric_catalog.pdf` | Fig. 1 |
| `results/temporal_trend_annotated.pdf` | Fig. 2 |
| `results/subreddit_personality_map.pdf` | Supplementary: sentiment by community and query axis (Section IV-B.2) |
| `results/surveillance_rto_trends.pdf` | Supplementary: employer-monitoring and RTO posts per year (Section IV-A.2) |
| `results/survey_adoption.pdf` | Supplementary: metrics and practices in use (Section IV-C) |
| `results/survey_relatedness.pdf` | Supplementary: answers to Q14 (mean 4.71) |
| `results/workflow.pdf` | Supplementary: study design. Hand-made drawing; its source is not in the repository |

Superseded scripts from earlier drafts are kept, for provenance only, on the
`legacy` branch (`scripts/legacy/`). They are not part of the reproduction path, and
two of them overwrite the published figures and table if run; see the README there.

---

## The statistical analysis

All of it lives in `analyze_corpus.py`. Messages are grouped into three metric
categories — **team-level metrics**, **individual-level metrics**, and **Scrum/Agile
process** — which the paper calls **metric orientation**; the code and its output call
them `framing` categories (`dora`, `individual`, `scrum`). Their terms are listed in
`QUERY_KEYWORDS.md`.

**Unadjusted tests** (descriptive only — they treat every message as independent,
which overstates the effective sample size because comments nest in threads):
chi-square over the three categories, the central team-vs-individual
2×2 contrast, and Kruskal–Wallis on the underlying compound-score distributions.
Reported alongside Cramér's V / φ, because with tens of thousands of messages a
trivial difference clears p < 0.05.

**Primary inference — logistic regression** of the binary negativity indicator
(VADER compound < −0.05) on the category:

```
neg ~ C(framing, Treatment('individual')) + C(subreddit) + C(year)
      + is_comment + log_len
```

- standard errors **clustered by thread** (Cameron & Miller 2015), because
  comments are not independent of the post they answer;
- fitted on the 32,682 messages matching **exactly one** category, so the
  category enters as a clean three-level factor (Table IV);
- a robustness fit on the full non-exclusive set with indicator variables;
- a posts-only contrast, where each message heads its own thread and no nesting
  exists;
- refitted with the RoBERTa label as the outcome, to show the association does
  not depend on VADER (Table VI, Section IV-B.1);
- refitted within each subreddit (same controls minus subreddit), with the
  per-community negativity gap, to show where the association comes from
  (Section IV-B.2).

The design is observational. Everything is reported as an **association**, never
as evidence that a metric's orientation *causes* the surrounding sentiment.

---

## Development

```bash
make verify    # py_compile + pyflakes over scripts/
make smoke     # synthetic end-to-end run
make clean     # drop docs/figures + generated CSVs, keep the corpus
make clean-mock  # drop data/mock/ only
```

`make clean` never touches `data/*_corpus.csv` or `data/survey_results.csv`.

Conventions: VADER negativity threshold is `< -0.05` everywhere; the three category
regexes are duplicated in `analyze_corpus.py` (`RX`) and `score_roberta.py` (`RX`)
and must be kept in sync — `score_roberta.py` only scores documents matching
them. `analyze_survey.py` addresses the survey export by column position and checks
the number of columns before computing anything.

`vaderSentiment` is pinned in `requirements.txt` because its lexicon determines every
sentiment score; the results were last verified with Python 3.13 and the versions
listed there.
