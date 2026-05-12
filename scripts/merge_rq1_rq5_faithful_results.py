#!/usr/bin/env python3
"""Merge RQ1/RQ5 faithful rerun outputs with split-job补缺 results.

The merge is key-based and non-destructive:
  - original seed directories are read first;
  - split directories are used only when the original key is missing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


RUN_ID = "20260507_015929"
DATASETS = [
    "Cora", "CiteSeer", "PubMed", "Computers", "Photo",
    "CoauthorCS", "CoauthorPhysics", "OGBNArxiv",
    "RomanEmpire", "AmazonRatings",
]
RQ1_REGIMES = ["both", "x_only", "a_only", "data_free"]
RQ1_ATTACKS = ["MEA2_Wu2022", "DFEA_I_RealGraph", "DFEA_II_E", "DFEA_III_E"]
RQ1_BUDGETS = [0.05, 0.10, 0.25, 0.50, 1.00]
RQ5_WM_DEFENSES = ["BackdoorWM", "RandomWM", "SurviveWM", "ImperceptibleWM", "Integrity"]
RQ5_INFO_DEFENSES = [
    "OutputPerturbation_low", "OutputPerturbation_high",
    "PredictionRounding_2bit", "PredictionRounding_top1",
    "PRADA", "AdaptiveMisinformation", "GradientRedirection",
]
RQ5_ATTACKS = RQ1_ATTACKS


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_bad_json": True, "_path": str(path), "_lineno": lineno, "_error": str(exc)}
                continue
            row["_source_file"] = str(path)
            yield row


def classify(row: dict) -> str:
    if row.get("track") == "RQ1":
        return "rq1"
    if "wm_acc_on_surrogate" in row or "surrogate_fidelity_to_defended" in row:
        return "rq5_watermark"
    if "fidelity_to_defended" in row or "fidelity_to_base" in row:
        return "rq5_info"
    return "unknown"


def key_for(row: dict):
    kind = classify(row)
    seed = int(row.get("seed", -1))
    if kind == "rq1":
        return (
            kind, seed, row.get("dataset"), row.get("regime"), row.get("attack"),
            round(float(row.get("budget", -1)), 8),
        )
    if kind in {"rq5_watermark", "rq5_info"}:
        return kind, seed, row.get("dataset"), row.get("defense"), row.get("attack")
    return kind, str(row)


def original_dirs(root: Path) -> list[Path]:
    return [
        root / f"rq1_rq5_faithful_{RUN_ID}",
        root / f"rq1_rq5_faithful_{RUN_ID}_seed1",
        root / f"rq1_rq5_faithful_{RUN_ID}_seed2",
    ]


def split_dirs(root: Path) -> list[Path]:
    return [
        root / f"rq1_rq5_faithful_{RUN_ID}_ogbn_wm_split",
        root / f"rq1_rq5_faithful_{RUN_ID}_coauthorphysics_wm_split",
    ]


def expected_keys():
    keys = {"rq1": set(), "rq5_watermark": set(), "rq5_info": set()}
    for seed in [0, 1, 2]:
        for ds in DATASETS:
            for regime in RQ1_REGIMES:
                for attack in RQ1_ATTACKS:
                    for budget in RQ1_BUDGETS:
                        keys["rq1"].add(("rq1", seed, ds, regime, attack, round(budget, 8)))
            for defense in RQ5_WM_DEFENSES:
                for attack in RQ5_ATTACKS:
                    keys["rq5_watermark"].add(("rq5_watermark", seed, ds, defense, attack))
            for defense in RQ5_INFO_DEFENSES:
                for attack in RQ5_ATTACKS:
                    keys["rq5_info"].add(("rq5_info", seed, ds, defense, attack))
    return keys


def clean_row(row: dict) -> dict:
    row = dict(row)
    row.pop("_source_file", None)
    return row


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(clean_row(row), sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("/home/kzhao2/GraphIPBench/outputs"))
    parser.add_argument("--outdir", type=Path, default=Path(f"/home/kzhao2/GraphIPBench/outputs/rq1_rq5_faithful_{RUN_ID}_merged"))
    args = parser.parse_args()

    merged = {}
    duplicates = Counter()
    bad_json = []

    for group in [original_dirs(args.outputs_root), split_dirs(args.outputs_root)]:
        for directory in group:
            for path in sorted(directory.rglob("*.jsonl")):
                for row in load_jsonl(path):
                    if row.get("_bad_json"):
                        bad_json.append(row)
                        continue
                    key = key_for(row)
                    if key in merged:
                        duplicates[key] += 1
                        continue
                    merged[key] = row

    rows_by_kind = defaultdict(list)
    for key, row in merged.items():
        rows_by_kind[key[0]].append(row)

    for rows in rows_by_kind.values():
        rows.sort(key=lambda r: json.dumps(key_for(r), sort_keys=True))

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "rq1.jsonl", rows_by_kind["rq1"])
    write_jsonl(args.outdir / "rq5_watermark.jsonl", rows_by_kind["rq5_watermark"])
    write_jsonl(args.outdir / "rq5_info.jsonl", rows_by_kind["rq5_info"])
    write_jsonl(args.outdir / "all.jsonl", rows_by_kind["rq1"] + rows_by_kind["rq5_watermark"] + rows_by_kind["rq5_info"])

    exp = expected_keys()
    present = {kind: {key for key in merged if key[0] == kind} for kind in exp}
    missing = {kind: sorted(exp[kind] - present[kind]) for kind in exp}

    missing_path = args.outdir / "missing_keys.tsv"
    with missing_path.open("w") as f:
        for kind, keys in missing.items():
            for key in keys:
                f.write("\t".join(map(str, key)) + "\n")

    summary = {
        "outdir": str(args.outdir),
        "counts": {kind: len(rows_by_kind[kind]) for kind in ["rq1", "rq5_watermark", "rq5_info"]},
        "expected": {"rq1": 2400, "rq5_watermark": 600, "rq5_info": 840, "all": 3840},
        "all_count": sum(len(rows_by_kind[kind]) for kind in ["rq1", "rq5_watermark", "rq5_info"]),
        "missing_counts": {kind: len(keys) for kind, keys in missing.items()},
        "duplicate_keys_ignored": len(duplicates),
        "duplicate_rows_ignored": sum(duplicates.values()),
        "bad_json_rows": len(bad_json),
        "missing_keys_file": str(missing_path),
    }
    with (args.outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

