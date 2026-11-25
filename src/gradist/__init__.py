"""gradist — the library behind mRouter: metadata + text + image -> tier -> model name.
Does not call any model.

See ../../ARCHITECTURE.md for how a decision is made and ../../design.md for the module, data
and API reference. `gradist.service` is the FastAPI service the mRouter console talks to.

Typical use:
    from gradist import load_tiers, RuleSet, resolve, validate_setup

    tiers = load_tiers("tiers.yaml")
    rules = RuleSet()  # or rules_from_file("rules.yaml")
    validate_setup(rules, tiers)  # once, at startup
    decision = resolve({"channel_type": "sports"}, "goal in the final", rules, tiers)
    # decision.model_name is what to call; this package never calls it.
"""

from gradist.catalog import CatalogEntry, ModelCatalog, load_catalog
from gradist.classifier import ClassifierError, EvalReport, TierClassifier, evaluate_classifier
from gradist.features import (
    Dinov2Embedder,
    Embedder,
    HashingEmbedder,
    ImageEmbedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
    build_feature_vector,
    load_image_bytes,
)
from gradist.labels import (
    LabelError,
    RawExample,
    TierLabel,
    export_for_human_judging,
    labels_from_rules,
    load_examples,
    load_human_judgments,
    merge_labels,
    stratified_sample,
    uncovered_examples,
)
from gradist.resolve import Decision, resolve, validate_setup
from gradist.rules import Rule, RuleError, RuleSet, rules_from_file
from gradist.tiers import TierSpec, TierTable, load_tiers

__version__ = "0.2.0"

__all__ = [
    "CatalogEntry",
    "ClassifierError",
    "Decision",
    "Dinov2Embedder",
    "Embedder",
    "EvalReport",
    "HashingEmbedder",
    "ImageEmbedder",
    "LabelError",
    "MetadataEncoder",
    "ModelCatalog",
    "RawExample",
    "Rule",
    "RuleError",
    "RuleSet",
    "SentenceTransformerEmbedder",
    "TierClassifier",
    "TierLabel",
    "TierSpec",
    "TierTable",
    "build_feature_vector",
    "evaluate_classifier",
    "export_for_human_judging",
    "labels_from_rules",
    "load_catalog",
    "load_examples",
    "load_human_judgments",
    "load_image_bytes",
    "load_tiers",
    "merge_labels",
    "resolve",
    "rules_from_file",
    "stratified_sample",
    "uncovered_examples",
    "validate_setup",
]
