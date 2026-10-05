# Archived scripts — superseded, do not run

These predate the current pipeline and were moved here on 2026-10-05. They are kept
only for provenance (the numbers quoted in earlier drafts of the TCC came from some
of them). **They are not part of the reproduction path and two of them silently
destroy published artifacts.**

Run `make help` for the supported pipeline.

## Why each one was archived

| Script | Superseded by | Why it is dangerous / wrong |
|---|---|---|
| `gerar_figuras_tese.py` | `analyze_corpus.py` | **Writes 5 of the 6 paper figures with the SAME filenames** (`metric_category_sentiment`, `subreddit_personality_map`, `temporal_trend_annotated`, `metric_catalog`, `surveillance_rto_trends`). Running it silently replaces the official figures with posts-only versions built on the legacy schema and the *un-curated* catalog. Also crashes at Fig. 4 if `data/metricas_catalogadas.csv` is missing, after having already written Figs 1–3. |
| `analise_sentimento.py` | `analyze_corpus.py` | **Writes `data/metricas_analise_sentimento.csv`, the paper's per-metric table, with a different column set** (adds a `Desvio Padrão` column) computed over posts-only from a different dataset. Silently replaces the published table. Also `NameError` if every metric subset comes back empty. |
| `collect_tcc_offline.py` | `build_corpus.py` | Top-10-per-query collector (2 orders of magnitude smaller than the paper corpus), 10 redundant decompression passes, no word-boundary matching (so it counts a *different* corpus), and it writes **plaintext `author` + `permalink`** — the exact privacy exposure `build_corpus.py` exists to prevent. |
| `collect_tcc_pullpush.py` | `build_corpus.py` | Hits `api.pullpush.io`, which is defunct. Unbounded pagination loop, naive (non-UTC) timestamps, plaintext author, 11+8 query terms vs the paper's 21+38. |
| `download_data.py` | `download_offline.py` | Downloads from an undocumented HTTP mirror with `verify=False` (TLS validation disabled) and **lowercases the output filenames**, so `build_corpus.py` cannot find `SoftwareEngineering_*` / `ExperiencedDevs_*` — 2 of the 5 subreddits vanish. Prints "Download concluído!" even when a file 404s, leaving a 0-byte `.zst`. |
| `download.sh` | `download_offline.py` | Hardcoded positional `--select` indices into one specific torrent metainfo. If the torrent is ever repacked, it selects the wrong files with exit code 0. Requires `npx webtorrent-cli`, which is declared nowhere. Its `-o` also double-nests the output dir. |
| `generate_torrent_cmd.py` | `download_offline.py` | Prints an **`aria2c`** command — a fourth download tool nothing else uses. Needs `bencode`, which was not in `requirements.txt`. Downloading the torrent metainfo happens at import time. Useful only for *inspecting* the torrent layout by hand. |
| `analyze_metrics.py` | `llm_catalog.py` + `consolidate_catalog.py` | Regex-only catalog prototype (45 hardcoded terms vs the curated pipeline's open LLM extraction). Its `--csv` output path has a double `.replace()` bug: `..._posts_unique.csv` becomes `..._posts_metrics_catalog_metrics_catalog.csv`. |
| `analise_metricas.py` | `analyze_corpus.py` | Same six metrics as the active per-metric section, but with **no word boundaries** (`loc` matches `local`/`block`/`clock`, `kpi`/`okr` match inside longer tokens), so its counts disagree with the paper's. Crashes with `ValueError: max() arg is an empty sequence` on the empty glob. |
| `gerar_resultados.py` | `analyze_corpus.py` | Three exploratory plots on the legacy schema. Mixes pyplot and OO APIs so `savefig` captures the *pandas* figure while the explicitly created one leaks; `plt.yscale('log')` silently drops the zero and negative Reddit scores; `palette=` without `hue=` is a seaborn deprecation. |

## If you must run one

Point it at a copy and expect different numbers:

```bash
python legacy/analyze_metrics.py data/<stamp>_corpus.csv        # inspect, do not trust
```

Never let anything in this directory write into `docs/figures/` or `data/`
without renaming its outputs first — that is exactly how the official
artifacts got clobbered before.
