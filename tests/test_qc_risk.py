from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gradist.features import HashingEmbedder
from gradist.qc_risk import QcRiskError, QcRiskHead
from gradist.resolve import resolve
from gradist.rules import Rule, RuleSet
from gradist.tiers import TierSpec, TierTable

TIERS = TierTable(
    tiers=(
        TierSpec("small", "", {"p": "m-small"}),
        TierSpec("medium", "", {"p": "m-medium"}),
        TierSpec("large", "", {"p": "m-large"}),
    ),
    default_provider="p",
    low_confidence_fallback="medium",
)
FIELDS = {"channel_name": ["Aaj Tak", "Ad Channel"], "program_type": ["Story Block", "Filler"]}
RISKY = {"channel_name": "Aaj Tak", "program_type": "Story Block"}
SAFE = {"channel_name": "Ad Channel", "program_type": "Filler"}
RISKY_TEXT = "live debate on election results with multiple guests and breaking updates"
SAFE_TEXT = "buy one get one free soap advertisement"


def _head() -> QcRiskHead:
    head = QcRiskHead(FIELDS, HashingEmbedder())
    meta = [RISKY] * 10 + [SAFE] * 10
    texts = [RISKY_TEXT] * 10 + [SAFE_TEXT] * 10
    head.fit_vectors(head.vectors(meta, texts), [True] * 10 + [False] * 10, escalate_share=0.5)
    return head


def test_risk_separates_fixed_from_clean() -> None:
    head = _head()
    assert head.risk(RISKY, RISKY_TEXT) > head.risk(SAFE, SAFE_TEXT)
    assert head.threshold is not None and head.risk(RISKY, RISKY_TEXT) >= head.threshold


def test_fit_needs_both_outcomes() -> None:
    head = QcRiskHead(FIELDS, HashingEmbedder())
    with pytest.raises(QcRiskError, match="both"):
        head.fit_vectors(head.vectors([SAFE, SAFE], [SAFE_TEXT, SAFE_TEXT]), [False, False])


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    head = _head()
    path = tmp_path / "risk.joblib"
    head.save(path)
    loaded = QcRiskHead.load(path, HashingEmbedder())
    assert loaded.threshold == head.threshold
    assert loaded.risk(RISKY, RISKY_TEXT) == pytest.approx(head.risk(RISKY, RISKY_TEXT))


def test_high_risk_sub_story_is_escalated() -> None:
    decision = resolve(RISKY, RISKY_TEXT, RuleSet(), TIERS, qc_risk=_head())
    assert decision.tier == "large"
    assert "escalated 'medium' to 'large'" in decision.reason


def test_low_risk_sub_story_is_left_alone() -> None:
    decision = resolve(SAFE, SAFE_TEXT, RuleSet(), TIERS, qc_risk=_head())
    assert decision.tier == "medium"
    assert "QC risk" not in decision.reason


def test_rule_is_never_escalated() -> None:
    rules = RuleSet(rules=(Rule.create({"program_type": "Story Block"}, "small"),))
    decision = resolve(RISKY, RISKY_TEXT, rules, TIERS, qc_risk=_head())
    assert decision.tier == "small"


def test_needs_channel_name_to_apply() -> None:
    meta = {"program_type": "Story Block"}
    decision = resolve(meta, RISKY_TEXT, RuleSet(), TIERS, qc_risk=_head())
    assert "QC risk" not in decision.reason


def test_vectors_shape_matches_fields_plus_text() -> None:
    head = QcRiskHead(FIELDS, HashingEmbedder())
    x = head.vectors([RISKY], [RISKY_TEXT])
    assert x.shape[0] == 1 and np.isfinite(x).all()
