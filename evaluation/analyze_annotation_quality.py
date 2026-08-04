#!/usr/bin/env python3
"""Report preference-label coverage, ambiguity, and inter-annotator agreement.

Run this on a directory containing one annotation export per annotator (or
subdirectories named by annotator).  The tool only compares labels on the same
bag, timestamp, and candidate pair, avoiding inflated agreement from unrelated
scenes.
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ABSTAIN = {404: "no_preference", 500: "both_bad"}


def _annotator_id(document, path, source):
    if source == "field":
        value = document.get("annotator_id") or document.get("annotator")
        if value is None:
            raise ValueError(f"{path}: missing annotator_id/annotator field")
        return str(value)
    if source == "auto":
        value = document.get("annotator_id") or document.get("annotator")
        if value is not None:
            return str(value)
    return path.parent.name


def _outcome(pair, choice):
    """Express choices in a canonical pair orientation."""
    pair = tuple(map(int, pair))
    choice = int(choice)
    if choice in ABSTAIN:
        return ABSTAIN[choice]
    if choice == min(pair):
        return "first"
    if choice == max(pair):
        return "second"
    return "invalid"


def load_votes(root, annotator_source):
    votes = defaultdict(dict)
    totals = Counter()
    for path in sorted(root.rglob("*.json")):
        document = json.loads(path.read_text())
        if "annotations_by_stamp" not in document:
            continue
        annotator = _annotator_id(document, path, annotator_source)
        bag = Path(document.get("bag", path.stem)).stem
        for stamp, annotation in document["annotations_by_stamp"].items():
            for entry in annotation.get("pairwise", []):
                pair = entry.get("pair")
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                canonical_pair = tuple(sorted(map(int, pair)))
                outcome = _outcome(pair, entry.get("choice", -1))
                if outcome == "invalid":
                    totals["invalid"] += 1
                    continue
                key = (bag, str(stamp), canonical_pair)
                votes[key][annotator] = outcome
                totals[outcome] += 1
    return votes, totals


def agreement(votes):
    observed, expected, comparisons = 0.0, 0.0, 0
    per_annotator = defaultdict(Counter)
    overlap_items = 0
    for labels in votes.values():
        if len(labels) < 2:
            continue
        overlap_items += 1
        for annotator, outcome in labels.items():
            per_annotator[annotator][outcome] += 1
        for left, right in combinations(labels.values(), 2):
            observed += left == right
            comparisons += 1
    if not comparisons:
        return {"overlap_items": 0, "comparisons": 0, "raw_agreement": None, "cohens_kappa_pooled": None}
    pooled = sum(per_annotator.values(), Counter())
    total = sum(pooled.values())
    expected = sum((count / total) ** 2 for count in pooled.values())
    raw = observed / comparisons
    kappa = (raw - expected) / (1 - expected) if expected < 1 else None
    return {"overlap_items": overlap_items, "comparisons": comparisons, "raw_agreement": raw, "cohens_kappa_pooled": kappa}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations_root", type=Path)
    parser.add_argument("--annotator-source", choices=("auto", "field", "parent"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    votes, totals = load_votes(args.annotations_root, args.annotator_source)
    report = {
        "unique_pairwise_items": len(votes),
        "labels": dict(totals),
        "abstention_rate": (totals["no_preference"] + totals["both_bad"]) / sum(totals.values()) if totals else None,
        "inter_annotator_agreement": agreement(votes),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
