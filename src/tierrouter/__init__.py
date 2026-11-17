"""llm-tier-router: metadata + text + image -> tier -> model name. Does not call any model.

See ../../ARCHITECTURE.md for the design and ../../PLAN.md for what is and is not built yet.
Tasks 1-8 are implemented, including real image features (Dinov2Embedder, Apache 2.0, see
features.py) and a FastAPI service (tierrouter.service) for the UI.

Typical use:
    from tierrouter import load_tiers, RuleSet, resolve, validate_setup

    tiers = load_tiers("tiers.yaml")
    rules = RuleSet()  # or rules_from_file("rules.yaml")
    validate_setup(rules, tiers)  # once, at startup
    decision = resolve({"channel_type": "sports"}, "goal in the final", rules, tiers)
    # decision.model_name is what to call; this package never calls it.
"""

from tierrouter.catalog import CatalogEntry, ModelCatalog, load_catalog
from tierrouter.classifier import ClassifierError, EvalReport, TierClassifier, evaluate_classifier
from tierrouter.features import (
    Dinov2Embedder,
    Embedder,
    HashingEmbedder,
    ImageEmbedder,
    MetadataEncoder,
    SentenceTransformerEmbedder,
    build_feature_vector,
    load_image_bytes,
)
from tierrouter.labels import (
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
from tierrouter.resolve import Decision, resolve, validate_setup
from tierrouter.rules import Rule, RuleError, RuleSet, rules_from_file
from tierrouter.tiers import TierSpec, TierTable, load_tiers

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
