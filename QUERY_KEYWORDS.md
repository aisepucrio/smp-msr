# Query keywords

This file lists the keywords used to retrieve Reddit posts and comments for the study (Section III-B, *Query filters*). The same lists are provided in machine-readable form in `query_keywords.csv`.

## Matching rules

- Matching is **case-insensitive**.
- Keywords are matched against the **full, untruncated text**: title and body for posts, body for comments.
- Short or ambiguous acronyms are matched **only as whole words**, so that, for example, `SLA` does not match "Slack": `RTO`, `OKR`, `KPI`, `DORA`, `SLA`, `SLO`, `SLI`, `MTTR`, `Jira`.
- All other terms, including multi-word phrases (e.g., `technical debt`), are matched as **substrings**.
- A message that matches terms from **both** groups is labeled with the mental-health axis. This precedence only decides the axis label. The metrics axis still keeps every message that does not also contain a mental-health term.
- The same rules are used for the terms that define the metric categories (see *Metric categories* below).

## Mental health (20 terms)

| Subgroup | Terms |
|---|---|
| Symptoms and care | `burnout`, `burned out`, `exhausted`, `overwhelmed`, `anxiety`, `stress`, `fear`, `depression`, `imposter syndrome`, `mental health`, `therapy`, `medication` |
| Working conditions | `overtime`, `quitting`, `micromanagement`, `work-life`, `toxic`, `hostile` |
| Workplace changes | `RTO`, `return to office` |

## Productivity measurement (33 terms)

| Subgroup | Terms |
|---|---|
| General measurement terms | `metric(s)`, `measure`, `measurement`, `productivity` |
| Delivery, reliability, and Agile metrics | `DORA`, `four key`, `deployment frequency`, `lead time`, `change failure`, `MTTR`, `cycle time`, `throughput`, `flow metric`, `velocity`, `story points`, `burndown`, `SLA`, `SLO`, `SLI` |
| Output and quality measures | `lines of code`, `pull request`, `code coverage`, `tech debt`, `technical debt` |
| Evaluation and goals | `performance review`, `OKR`, `KPI`, `benchmark`, `tracking` |
| Process and tools | `scrum`, `sprint`, `kanban`, `Jira` |

## Metric categories (sentiment analysis, Section IV-B)

Messages are grouped into three metric categories, which define the rows of Tables III, IV and VI. The terms are the regexes `RX` in `scripts/analyze_corpus.py` (duplicated in `scripts/score_roberta.py`). The categories are not mutually exclusive.

| Category | Label in the code | Terms |
|---|---|---|
| Team-level metrics | `dora` | `DORA`\*, `deployment frequency`, `change failure`, `mean time to recover`, `four key` |
| Individual-level metrics | `individual` | `PIP`\*, `performance improvement plan`, `performance review`, `story point(s)`\*, `velocity`\*, `OKR(s)`\*, `KPI(s)`\*, `lines of code` |
| Scrum/Agile process | `scrum` | `scrum`\*, `agile`\* |

\* matched as a whole word; the other terms are matched as substrings.

Table V uses a separate, looser per-metric matching (the `METRICS` dictionary in `scripts/analyze_corpus.py`): for example, its *Throughput / DORA* row counts any message that names throughput or DORA.

## Notes

- The subgroups organize the lists for reading. They are not used in matching: a message is assigned only to a query group (mental health or productivity).
- `metric(s)` stands for both `metric` and `metrics`.
