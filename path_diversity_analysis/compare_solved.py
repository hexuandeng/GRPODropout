"""Compare, per question, whether different sampling sizes (e.g. k32 vs k256) for the
same (model, dataset) solve the same questions.

"Solved" defaults to the pass@k criterion: at least one of the question's N rollouts is
correct (is_correct / any of answers_correctness). The solve rate (fraction of correct
rollouts) is also reported to reveal deeper differences.

Usage:
    python compare_solved.py fileA.jsonl fileB.jsonl [--labels k32,k256]
    # or give a directory to auto-match all test_*_k<N>_*.jsonl inside it and compare pairwise
    python compare_solved.py /path/to/aime2025
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple


def _load(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r.get("id", idx))
            corr = r.get("answers_correctness") or []
            corr = [bool(x) for x in corr]
            n = len(corr)
            nc = sum(1 for x in corr if x)
            solved = bool(r.get("is_correct")) if "is_correct" in r else (nc > 0)
            out[qid] = {
                "solved": solved,           # pass@k
                "num_correct": nc,
                "n": n,
                "solve_rate": (nc / n) if n else 0.0,
                "gold": str(r.get("gold_answer", "")),
            }
    return out


def _k_of(path: str) -> str:
    m = re.search(r"_k(\d+)_", os.path.basename(path))
    return f"k{m.group(1)}" if m else os.path.basename(path)


def compare(path_a: str, path_b: str,
            label_a: Optional[str] = None,
            label_b: Optional[str] = None) -> dict:
    a = _load(path_a)
    b = _load(path_b)
    la = label_a or _k_of(path_a)
    lb = label_b or _k_of(path_b)

    common = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else x)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    solved_a = {i for i in a if a[i]["solved"]}
    solved_b = {i for i in b if b[i]["solved"]}
    both = sorted((solved_a & solved_b) & set(common), key=lambda x: int(x) if x.isdigit() else x)
    a_not_b = sorted((solved_a - solved_b) & set(common), key=lambda x: int(x) if x.isdigit() else x)
    b_not_a = sorted((solved_b - solved_a) & set(common), key=lambda x: int(x) if x.isdigit() else x)

    gold_mismatch = [i for i in common if a[i]["gold"] != b[i]["gold"] and a[i]["gold"] and b[i]["gold"]]

    same_solved_set = (solved_a & set(common)) == (solved_b & set(common))

    per_q = []
    for i in common:
        per_q.append({
            "id": i,
            f"{la}_solved": a[i]["solved"], f"{la}_correct": f"{a[i]['num_correct']}/{a[i]['n']}",
            f"{la}_solve_rate": round(a[i]["solve_rate"], 4),
            f"{lb}_solved": b[i]["solved"], f"{lb}_correct": f"{b[i]['num_correct']}/{b[i]['n']}",
            f"{lb}_solve_rate": round(b[i]["solve_rate"], 4),
            "solved_agrees": a[i]["solved"] == b[i]["solved"],
        })

    return {
        "label_a": la, "label_b": lb,
        "path_a": path_a, "path_b": path_b,
        "n_common": len(common), "only_a": only_a, "only_b": only_b,
        "gold_mismatch": gold_mismatch,
        f"n_solved_{la}": len(solved_a & set(common)),
        f"n_solved_{lb}": len(solved_b & set(common)),
        "solved_in_both": both,
        f"solved_only_{la}": a_not_b,
        f"solved_only_{lb}": b_not_a,
        "same_solved_set": same_solved_set,
        "per_question": per_q,
    }


def print_report(res: dict):
    la, lb = res["label_a"], res["label_b"]
    print(f"\n{'='*66}")
    print(f"Compare: {la}  vs  {lb}")
    print(f"  A: {res['path_a']}")
    print(f"  B: {res['path_b']}")
    print(f"{'='*66}")
    print(f"Common questions: {res['n_common']}"
          + (f" | only in A: {res['only_a']}" if res['only_a'] else "")
          + (f" | only in B: {res['only_b']}" if res['only_b'] else ""))
    if res["gold_mismatch"]:
        print(f"[warn] questions with mismatched gold answers: {res['gold_mismatch']}")
    else:
        print("Gold answers: all consistent [ok]")
    print()
    print(f"Solved (pass@k) counts: {la}={res[f'n_solved_{la}']}  {lb}={res[f'n_solved_{lb}']}")
    print(f"Solved by both: {len(res['solved_in_both'])} -> {res['solved_in_both']}")
    print(f"Solved only by {la}: {res[f'solved_only_{la}']}")
    print(f"Solved only by {lb}: {res[f'solved_only_{lb}']}")
    print(f"==> Are the solved sets identical: {'yes [ok]' if res['same_solved_set'] else 'no [x]'}")
    print()
    print(f"{'id':>4} | {la+' solved':>12} {la+' correct':>12} | {lb+' solved':>12} {lb+' correct':>12} | agree")
    print("-" * 78)
    for row in res["per_question"]:
        flag = "" if row["solved_agrees"] else "  <== diff"
        print(f"{row['id']:>4} | "
              f"{str(row[f'{la}_solved']):>12} {row[f'{la}_correct']:>12} | "
              f"{str(row[f'{lb}_solved']):>12} {row[f'{lb}_correct']:>12} | "
              f"{str(row['solved_agrees']):>5}{flag}")


def _find_jsonls(d: str) -> List[str]:
    import glob
    cands = glob.glob(os.path.join(d, "test_*_k*_*.jsonl"))
    return sorted(cands, key=lambda p: os.path.getsize(p))


def main():
    ap = argparse.ArgumentParser(description="Compare, per question, whether different sampling sizes solve the same questions")
    ap.add_argument("inputs", nargs="+", help="two jsonl files, or one directory containing them")
    ap.add_argument("--labels", default=None, help="two comma-separated labels, e.g. k32,k256")
    ap.add_argument("--save", default=None, help="save the result JSON to this path")
    args = ap.parse_args()

    if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        files = _find_jsonls(args.inputs[0])
        if len(files) < 2:
            print(f"Matched {len(files)} test_*_k*.jsonl in the directory; at least 2 are needed")
            return
        path_a, path_b = files[0], files[1]
    elif len(args.inputs) >= 2:
        path_a, path_b = args.inputs[0], args.inputs[1]
    else:
        print("Provide two jsonl files, or one directory")
        return

    la = lb = None
    if args.labels:
        parts = [x.strip() for x in args.labels.split(",")]
        if len(parts) == 2:
            la, lb = parts

    res = compare(path_a, path_b, la, lb)
    print_report(res)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
