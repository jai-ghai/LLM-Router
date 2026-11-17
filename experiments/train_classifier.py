"""Train a TierClassifier from rule-derived + human-judged labels, then report its fit.

Generic: takes any tiers.yaml / rules.yaml / examples.jsonl, nothing here is BARC-specific. See
experiments/barc_test/ for one worked example built from Minfy's BARC production code (read-only).

Usage:
    python experiments/train_classifier.py --tiers TIERS.yaml --examples EXAMPLES.jsonl \
        [--rules RULES.yaml] [--human-judgments JUDGED.jsonl] [--save CLASSIFIER.joblib] \
        [--baseline-tier TIER] [--eval-field FIELD]
"""

from __future__ import annotations

import argparse
import sys

from tierrouter import (
    HashingEmbedder,
    MetadataEncoder,
    RuleSet,
    TierClassifier,
    evaluate_classifier,
    labels_from_rules,
    load_examples,
    load_human_judgments,
    load_tiers,
    merge_labels,
    rules_from_file,
    uncovered_examples,
)
from tierrouter.labels import RawExample


def infer_fields(examples: list[RawExample]) -> dict[str, list[str]]:
    """Every metadata field and value seen in `examples`. A convenience for this script only —

    the library itself never infers fields; a real deployment should declare them explicitly so
    a value that stops appearing in training data does not silently become "unknown" later.
    """
    fields: dict[str, set[str]] = {}
    for example in examples:
        for key, value in example.metadata.items():
            fields.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in fields.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tiers", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--rules", default=None)
    parser.add_argument("--human-judgments", default=None)
    parser.add_argument("--save", default=None, help="path to save the trained classifier (joblib)")
    parser.add_argument("--baseline-tier", default=None, help="default: tiers.yaml's low_confidence_fallback")
    parser.add_argument("--eval-field", default=None, help="default: the first declared metadata field")
    args = parser.parse_args()

    tiers = load_tiers(args.tiers)
    rules = rules_from_file(args.rules) if args.rules else RuleSet()
    rules.validate(tiers.names)

    examples = load_examples(args.examples)
    by_id = {example.example_id: example for example in examples}

    rule_labels = labels_from_rules(examples, rules)
    human_labels = (
        load_human_judgments(args.human_judgments, by_id, known_tiers=tiers.names)
        if args.human_judgments
        else []
    )
    labels = merge_labels(rule_labels, human_labels)
    if not labels:
        sys.exit(
            "No labels at all: no rule matched anything, and no --human-judgments were given.\n"
            "Export the uncovered examples for human judging first (tierrouter.export_for_human_judging)."
        )

    fields = infer_fields(examples)
    encoder = MetadataEncoder(fields)
    embedder = HashingEmbedder()  # swap for SentenceTransformerEmbedder("BAAI/bge-m3") with real text
    classifier = TierClassifier(encoder, embedder)
    classifier.fit(labels)

    baseline_tier = args.baseline_tier or tiers.low_confidence_fallback
    eval_field = args.eval_field or next(iter(fields))
    report = evaluate_classifier(classifier, labels, baseline_tier=baseline_tier, metadata_field=eval_field)

    rule_count = sum(1 for label in labels if label.source == "rule")
    human_count = sum(1 for label in labels if label.source == "human")
    print(f"Labels used: {len(labels)} total ({rule_count} from rules, {human_count} from human judgments)")
    print(f"Declared metadata fields: { {k: len(v) for k, v in fields.items()} }")
    print(f"Accuracy on this labeled set: {report.accuracy:.2f}")
    print(f"Baseline (always '{baseline_tier}'): {report.baseline_accuracy:.2f}")
    print(f"Per-'{eval_field}' bucket accuracy: {report.accuracy_by_bucket}")
    print(f"Bucket sizes: {report.bucket_sizes}")
    print()
    print(
        "CAVEAT: this evaluates on the SAME examples it trained on, because there are too few "
        "labels for an honest held-out split. This only confirms the code runs end to end — it "
        "is not a quality measurement. Do not use this number to decide anything (ARCHITECTURE.md §8)."
    )

    uncovered = uncovered_examples(examples, rules)
    if uncovered:
        print(f"\n{len(uncovered)} example(s) no rule reaches: {[e.example_id for e in uncovered]}")
        print("These are the candidates for a human-judged golden set (strategy 2).")

    if args.save:
        classifier.save(args.save)
        print(f"\nSaved the trained classifier to {args.save}")


if __name__ == "__main__":
    main()
