"""
Selectively download the submissions (.zst) for the five target subreddits from the
Academic Torrents Reddit dump, using libtorrent. Submissions only (~206 MB total).

    python scripts/download_offline.py             # submissions only
    python scripts/download_offline.py --comments  # also fetch comments (~1.5 GB total)

Files land in data/reddit/subreddits24/, ready for scripts/build_corpus.py.
"""
import argparse
import os
import time
import urllib.request

import libtorrent as lt

TORRENT_URL = "https://academictorrents.com/download/1614740ac8c94505e4ecb9d88be8bed7b6afddd4.torrent"
OUT_DIR = "data/reddit"  # torrent paths already include subreddits24/
SUBS = ["cscareerquestions", "experienceddevs", "womenintech",
        "softwareengineering", "devops"]


def wanted(name, kinds):
    n = name.lower().split("/")[-1]
    return any(n.startswith(f"{s}_{k}") for s in SUBS for k in kinds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", action="store_true", help="also download comments")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="save_path for the torrent (default: %(default)s; the "
                         "torrent's internal paths already include subreddits24/)")
    args = ap.parse_args()
    kinds = ["submissions"] + (["comments"] if args.comments else [])

    os.makedirs(args.out_dir, exist_ok=True)
    print("Fetching torrent metadata...")
    req = urllib.request.Request(TORRENT_URL, headers={"User-Agent": "Mozilla/5.0"})
    torrent_bytes = urllib.request.urlopen(req, timeout=60).read()
    info = lt.torrent_info(lt.bdecode(torrent_bytes))

    fs = info.files()
    priorities, selected, total = [], [], 0
    for i in range(info.num_files()):
        path = fs.file_path(i)
        if wanted(path, kinds):
            priorities.append(4)
            selected.append((path, fs.file_size(i)))
            total += fs.file_size(i)
        else:
            priorities.append(0)

    if not selected:
        raise SystemExit(
            f"No files in the torrent matched {SUBS} x {kinds}. The torrent layout "
            f"has probably changed; inspect it with scripts/legacy/generate_torrent_cmd.py "
            f"instead of guessing --select indices.")

    print(f"Selecting {len(selected)} files ({total/1048576:.0f} MB):")
    for p, sz in selected:
        print(f"  {p.split('/')[-1]:42s} {sz/1048576:8.1f} MB")

    ses = lt.session({"listen_interfaces": "0.0.0.0:6881"})
    h = ses.add_torrent({
        "ti": info,
        "save_path": args.out_dir,
        "file_priorities": priorities,
    })
    print("\nDownloading (Ctrl-C to stop)...")
    while True:
        s = h.status()
        if s.total_wanted and s.total_wanted_done >= s.total_wanted:
            break
        if s.is_seeding:  # torrent already complete in this save_path
            break
        want = s.total_wanted / 1048576
        done = s.total_wanted_done / 1048576
        pct = (done / want * 100) if want else 0.0
        print(f"\r  {pct:5.1f}%  {done:7.1f}/{want:.1f} MB  "
              f"down {s.download_rate/1024:6.0f} kB/s  peers {s.num_peers}",
              end="", flush=True)
        time.sleep(2)
    print("\nDone. Files in", os.path.join(args.out_dir, "subreddits24"))
    print("Next: python scripts/build_corpus.py --comments")


if __name__ == "__main__":
    main()
