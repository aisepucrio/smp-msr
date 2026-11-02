"""
Build the publishable, pseudonymized release of the working dataset.

The working corpus (build_corpus.py) is pseudonymized but NOT safe to publish: author
hashes are unsalted SHA-256 prefixes (reversible by hashing candidate usernames) and the
Reddit post/comment ids rebuild each message's URL. This script replaces both with keyed
hashes (HMAC-SHA-256) under a secret key that is never distributed, so the release keeps
the thread structure (which comment belongs to which post, which messages share an
author) without pointing back to Reddit.

    python scripts/pseudonymize.py --key-file ~/secrets/smp-msr.key --out-dir release/

What it does
  * author_hash  -> HMAC(key, "author:" + author_hash)[:16]   ("[deleted]" -> "deleted")
  * post id      -> HMAC(key, "post:" + id)[:16]; a comment's link_id gets the same
                    mapping, so threads stay linked
  * comment id   -> HMAC(key, "comment:" + id)[:16]
  * created_utc  -> date only (YYYY-MM-DD); the analysis uses year and month
  * title/body   -> letters and digits of "u/username" mentions and of reddit.com /
                    redd.it / "/r/<sub>/comments/..." link paths are masked with "x"
                    (the subreddit name is kept); length and punctuation are unchanged,
                    so every number in the paper is reproduced exactly from the release
  * RoBERTa labels, LLM catalog labels and the manual review of employer-monitoring
    posts are re-keyed with the same mapping, so
    analyze_corpus.py runs on the release unchanged
  * survey       -> the optional e-mail column is emptied (kept, so column positions hold)
  * the curated / full catalogs and the per-metric table carry no identifiers and are
    copied as they are

The key file is created (32 random bytes, hex) if it does not exist. Keep it OUTSIDE the
repository and out of the release; it is only needed to produce a later release with the
same pseudonyms. Losing it is harmless for reproduction.

Residual risk: message text is kept verbatim (it is the object of study), so a determined
reader can still find a message by searching Reddit for its text. The release is
pseudonymized, not anonymized.
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted")

# Masks replace letters and digits with "x" and keep every other character, so the
# text keeps its length and punctuation (VADER reads "!" and "?", the regression uses
# text length) and the subreddit named in a link stays readable.
USER_RX = re.compile(r"(?<![\w/])(/?u/)([A-Za-z0-9_-]{3,20})\b")
ABS_LINK_RX = re.compile(
    r"((?:https?://)?(?:[a-z0-9-]+\.)?(?:[a-z]*ddit\.com|redd\.it)(?:/r/[A-Za-z0-9_]+)?)(\S*)", re.I)
REL_LINK_RX = re.compile(r"(?<![\w.])(/?r/[A-Za-z0-9_]+)(/comments/\S*)")
DELETED_AUTHOR = hashlib.sha256(b"[deleted]").hexdigest()[:16]


def mask(s):
    return re.sub(r"[A-Za-z0-9]", "x", s)


def load_key(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32) + "\n")
        os.chmod(path, 0o600)
        print(f"[key] new key written to {path} — keep it private, never commit it")
    return bytes.fromhex(open(path).read().strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="data/20260607_203437_corpus.csv")
    ap.add_argument("--roberta", default="data/roberta_sentiment.jsonl")
    ap.add_argument("--llm-labels", default="data/llm_labels.jsonl")
    ap.add_argument("--survey", default="data/survey_results.csv")
    ap.add_argument("--surveillance-review", default="data/surveillance_manual_review.csv",
                    help="manually reviewed employer-monitoring posts (re-keyed if present)")
    ap.add_argument("--catalogs", nargs="*", default=[
        "data/metricas_catalogadas_filtradas.csv", "data/metricas_catalogadas.csv",
        "data/metricas_analise_sentimento.csv"])
    ap.add_argument("--key-file", required=True, help="secret key (created if missing)")
    ap.add_argument("--out-dir", default="release")
    a = ap.parse_args()

    key = load_key(a.key_file)
    os.makedirs(a.out_dir, exist_ok=True)

    def h(kind, value):
        return hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:16]

    def scrub(text):
        if not isinstance(text, str) or not text:
            return text
        text = ABS_LINK_RX.sub(lambda m: m.group(1) + mask(m.group(2)), text)
        text = REL_LINK_RX.sub(lambda m: m.group(1) + mask(m.group(2)), text)
        return USER_RX.sub(lambda m: m.group(1) + mask(m.group(2)), text)

    # ---- corpus ----
    df = pd.read_csv(a.corpus, dtype={"id": str, "link_id": str, "author_hash": str,
                                      "title": str, "selftext": str}, low_memory=False)
    body = df["title"].fillna("") + " " + df["selftext"].fillna("")
    n_users = body.str.contains(USER_RX).sum()
    n_links = (body.str.contains(ABS_LINK_RX) | body.str.contains(REL_LINK_RX)).sum()
    is_post = df["type"] == "post"
    df["id"] = [h("post" if p else "comment", i) for p, i in zip(is_post, df["id"])]
    df["link_id"] = [h("post", l) if (not p and isinstance(l, str) and l) else ""
                     for p, l in zip(is_post, df["link_id"])]
    df["author_hash"] = ["deleted" if x == DELETED_AUTHOR else h("author", x)
                         for x in df["author_hash"]]
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["title"] = df["title"].map(scrub)
    df["selftext"] = df["selftext"].map(scrub)
    out_corpus = os.path.join(a.out_dir, "corpus_pseudonymized.csv.gz")
    # CRLF like build_corpus.py's csv writer: fields holding a lone "\r" then get quoted;
    # with a bare "\n" terminator they are not, and the row splits on re-reading.
    df.to_csv(out_corpus, index=False, compression="gzip", lineterminator="\r\n")
    print(f"[corpus] {len(df):,} messages -> {out_corpus} "
          f"(user mentions scrubbed in {n_users:,}, reddit links in {n_links:,})")

    # ---- RoBERTa labels (keys are "post:<id>" / "comment:<id>") ----
    with open(a.roberta) as fi, open(os.path.join(a.out_dir, "roberta_sentiment.jsonl"), "w") as fo:
        for line in fi:
            if line.strip():
                r = json.loads(line)
                kind, _, rid = r["key"].partition(":")
                r["key"] = f"{kind}:{h(kind, rid)}"
                fo.write(json.dumps(r) + "\n")
    print("[roberta] re-keyed")

    # ---- LLM catalog labels (one record per post) ----
    with open(a.llm_labels, encoding="utf-8") as fi, \
            open(os.path.join(a.out_dir, "llm_labels.jsonl"), "w", encoding="utf-8") as fo:
        for line in fi:
            if line.strip():
                r = json.loads(line)
                r["id"] = h("post", r["id"])
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[llm labels] re-keyed")

    # ---- manual review of employer-monitoring posts (post ids re-keyed) ----
    if os.path.exists(a.surveillance_review):
        rv = pd.read_csv(a.surveillance_review, dtype={"id": str})
        rv["id"] = [h("post", i) for i in rv["id"]]
        keep = [c for c in ("id", "subreddit", "title", "employer_monitoring") if c in rv]
        rv[keep].to_csv(os.path.join(a.out_dir, "surveillance_manual_review.csv"), index=False)
        print(f"[surveillance review] {len(rv)} reviewed posts re-keyed")

    # ---- survey: blank the optional e-mail column (kept, empty: analyze_survey.py
    # addresses the export by column position, so dropping it would shift the rest) ----
    sv = pd.read_csv(a.survey)
    email_cols = [c for c in sv.columns if "email" in c.lower() or "e-mail" in c.lower()]
    sv[email_cols] = ""
    sv.to_csv(os.path.join(a.out_dir, "survey_results.csv"), index=False)
    print(f"[survey] {len(sv)} responses, blanked {email_cols}")

    # ---- identifier-free tables ----
    for c in a.catalogs:
        shutil.copy(c, a.out_dir)
    print(f"[tables] copied {len(a.catalogs)}")

    # ---- checksums ----
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for f in sorted(os.listdir(a.out_dir)):
            p = os.path.join(a.out_dir, f)
            if os.path.isfile(p) and f != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {f}\n")
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
