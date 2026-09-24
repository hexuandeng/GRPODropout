#!/usr/bin/env python3
"""Split 100 non-math MMLU-Pro questions, evenly across the 13 non-math categories.

Reads the existing eval/data/MMLU-Pro.jsonl (12032 rows) — no re-download.
Writes eval/heldout/data/MMLU-Pro-100-nomath.jsonl (kept separate; original untouched).

Row schema is preserved exactly as the source (id/question/choices/answer/
source/category/src), so eval's parser (mmlu_pro is a CHOICE dataset) works
unchanged.
"""

import json
import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# eval/heldout/  -> eval/data/MMLU-Pro.jsonl
SRC = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "MMLU-Pro.jsonl"))
OUT_DIR = os.path.join(BASE_DIR, "data")
OUT = os.path.join(OUT_DIR, "MMLU-Pro-100-nomath.jsonl")

SEED = 42
TOTAL = 100
EXCLUDE = {"math"}  # drop the math category (in-domain for the model)


def main():
    random.seed(SEED)
    by_cat = defaultdict(list)
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cat = d.get("category", "?")
            if cat in EXCLUDE:
                continue
            by_cat[cat].append(d)

    cats = sorted(by_cat.keys())
    n_cats = len(cats)
    base = TOTAL // n_cats          # even quota per category
    remainder = TOTAL - base * n_cats

    # deterministic shuffle within each category, then take the quota
    picked = []
    quota = {c: base for c in cats}
    # distribute the remainder to the largest categories (deterministic order)
    for c in sorted(cats, key=lambda c: (-len(by_cat[c]), c))[:remainder]:
        quota[c] += 1

    for c in cats:
        pool = by_cat[c][:]
        random.shuffle(pool)
        take = min(quota[c], len(pool))
        picked.extend(pool[:take])

    # if any category was too small to fill its quota, top up from the rest
    if len(picked) < TOTAL:
        picked_ids = {id(x) for x in picked}
        leftovers = [x for c in cats for x in by_cat[c] if id(x) not in picked_ids]
        random.shuffle(leftovers)
        picked.extend(leftovers[: TOTAL - len(picked)])

    picked = picked[:TOTAL]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    dist = defaultdict(int)
    for d in picked:
        dist[d.get("category", "?")] += 1
    print(f"Wrote {len(picked)} rows -> {OUT}")
    print("Per-category counts:")
    for c in sorted(dist):
        print(f"  {dist[c]:3d}  {c}")


if __name__ == "__main__":
    main()
