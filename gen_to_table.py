#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-robust parser for fairseq-generate outputs.

- Tolerates spaces or tabs between fields
- Ignores logging prefixes (timestamps, [INFO], etc.)
- Handles extra tokens after tags (e.g., language markers)
- Selects first H- by default; --prefer-best picks best D- score
- Can add image column via split-list + image-root

Output: .csv / .tsv / .md with columns:
  id, source, reference, hypothesis, [score], [image]
"""
import re, argparse, pathlib, csv
from typing import Dict, Tuple, List, Optional

# Matches lines even with prefixes:
# e.g. "[INFO] 12:00:00 S-478 ...", "S-  478 ...", "   S-478\t..."
TAG_RE = re.compile(r"""
    ^\s*                         # leading spaces
    (?:\[?[A-Za-z0-9:_\-\./ ]+\]?\s+)*?   # optional noisy prefixes
    ([STHD])\s*-\s*(\d+)         # group(1)=tag, group(2)=id
    \s+                          # at least one whitespace before payload
    (.*)$                        # payload
""", re.VERBOSE)

def parse_generate(path: str, prefer_best_by_d: bool) -> Tuple[Dict[int,str], Dict[int,str], Dict[int,str], Dict[int,float]]:
    S: Dict[int,str] = {}
    T: Dict[int,str] = {}
    H: Dict[int,str] = {}
    Dscore: Dict[int,float] = {}
    H_cands: Dict[int, List[Tuple[float, str]]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            ln = raw.strip("\n\r")
            if not ln:
                continue
            m = TAG_RE.match(ln)
            if not m:
                continue
            tag, sid, payload = m.group(1), int(m.group(2)), m.group(3).strip()

            if tag == "S":
                # Source: whole payload is the sentence
                S[sid] = payload
            elif tag == "T":
                # Reference
                T[sid] = payload
            elif tag == "H":
                # Hypothesis: may look like "<score> <text>" or just "<text>"
                # Try to peel off first token as score if float-like
                parts = payload.split(None, 1)
                if parts:
                    if len(parts) == 2 and _is_float(parts[0]):
                        hyp = parts[1].strip()
                    else:
                        hyp = payload
                    if not prefer_best_by_d and sid not in H:
                        H[sid] = hyp
            elif tag == "D":
                # D-: "<score> <text>" (standard)
                parts = payload.split(None, 1)
                if parts:
                    if _is_float(parts[0]):
                        sc = float(parts[0])
                        hyp = parts[1].strip() if len(parts) == 2 else ""
                    else:
                        # Some exotic formats; fallback
                        sc = float("-inf")
                        hyp = payload
                    if prefer_best_by_d:
                        H_cands.setdefault(sid, []).append((sc, hyp))
    if prefer_best_by_d:
        for i, cands in H_cands.items():
            best = max(cands, key=lambda x: x[0])
            Dscore[i] = best[0]
            H[i] = best[1]
    return S, T, H, Dscore

def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def load_split_list(split_list_file: Optional[str]) -> Dict[int, str]:
    if not split_list_file: return {}
    mp = {}
    with open(split_list_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, ln in enumerate(f):
            fn = ln.strip()
            if fn: mp[idx] = fn
    return mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="fairseq-generate output file (e.g., results/test2016.out)")
    ap.add_argument("--out", required=True, help="output path: .csv / .tsv / .md")
    ap.add_argument("--ids", help="comma-separated ids to export (e.g., 0,12,57). Default: all sorted ids in file")
    ap.add_argument("--prefer-best", action="store_true", help="select best hypothesis by D- score instead of first H-")
    ap.add_argument("--split-list", help="e.g., flickr30k/test_2016_flickr.txt (line idx = sample id)")
    ap.add_argument("--image-root", help="prefix dir for images, joined with split-list names")
    ap.add_argument("--include-score", action="store_true", help="add a score column if available")
    args = ap.parse_args()

    S, T, H, Dscore = parse_generate(args.gen, args.prefer_best)

    # Compose id set
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = sorted(set(S) | set(T) | set(H))
    if not ids:
        # print small hint with a quick sample of lines
        raise SystemExit("No ids found. Make sure your file has lines like 'S-0 ...', 'T-0 ...', 'H-0 ...' (even with prefixes).")

    # Image mapping
    id2img = {}
    if args.split_list:
        split_map = load_split_list(args.split_list)
        root = pathlib.Path(args.image_root) if args.image_root else pathlib.Path(".")
        for i in ids:
            if i in split_map:
                id2img[i] = str((root / split_map[i]).as_posix())

    out_path = pathlib.Path(args.out)
    ext = out_path.suffix.lower()

    # Build rows
    rows = []
    for i in ids:
        row = {
            "id": i,
            "source": S.get(i, ""),
            "reference": T.get(i, ""),
            "hypothesis": H.get(i, "")
        }
        if args.include_score and i in Dscore:
            row["score"] = Dscore[i]
        if i in id2img:
            row["image"] = id2img[i]
        rows.append(row)

    # Write
    if ext in (".csv", ".tsv"):
        fieldnames = ["id", "source", "reference", "hypothesis"]
        if args.include_score: fieldnames.append("score")
        if any("image" in r for r in rows): fieldnames.append("image")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=("\t" if ext==".tsv" else ","))
            writer.writeheader()
            for r in rows: writer.writerow(r)
    elif ext == ".md":
        cols = ["id", "source", "reference", "hypothesis"]
        if args.include_score: cols.append("score")
        if any("image" in r for r in rows): cols.append("image")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"]*len(cols)) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    else:
        raise SystemExit("Please set --out to .csv / .tsv / .md")

    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
