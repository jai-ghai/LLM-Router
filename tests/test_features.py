from __future__ import annotations

import numpy as np
import pytest

from gradist.features import HashingEmbedder, MetadataEncoder, build_feature_vector


class FakeImageEmbedder:
    """A fixed, deterministic fake — no real model, for fast unit tests of the plumbing around
    an image embedder. Dinov2Embedder itself is covered by test_dinov2_embedder.py, gated behind
    an env var since it downloads a real model."""

    width = 4

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def embed(self, images: list[bytes]) -> np.ndarray:
        self.calls.extend(images)
        return np.array([[float(len(img)), 1.0, 2.0, 3.0] for img in images])


def test_known_value_sets_its_column() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "sports"]})
    vec = enc.encode({"channel_type": "sports"})
    assert vec.tolist() == [0.0, 1.0, 0.0]  # news, sports, unknown


def test_unseen_value_goes_to_unknown_not_an_error() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "sports"]})
    vec = enc.encode({"channel_type": "weather"})
    assert vec.tolist() == [0.0, 0.0, 1.0]


def test_missing_field_in_input_goes_to_unknown() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "sports"]})
    vec = enc.encode({})
    assert vec.tolist() == [0.0, 0.0, 1.0]


def test_undeclared_field_name_raises() -> None:
    enc = MetadataEncoder({"channel_type": ["news"]})
    with pytest.raises(KeyError):
        enc.encode({"anchor": "x"})


def test_duplicate_known_values_are_deduped() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "news", "sports"]})
    assert enc.width == 3  # news, sports, unknown — not 4


def test_multiple_fields_concatenate_independently() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "sports"], "anchor": ["a", "b"]})
    vec = enc.encode({"channel_type": "news", "anchor": "z"})
    assert vec.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # news col set, anchor -> unknown


def test_no_fields_at_all_is_rejected() -> None:
    with pytest.raises(ValueError):
        MetadataEncoder({})


def test_hashing_embedder_shape_and_determinism() -> None:
    embedder = HashingEmbedder(n_features=64)
    a = embedder.embed(["hello world"])
    b = embedder.embed(["hello world"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)


def test_build_feature_vector_concatenates_metadata_and_text() -> None:
    enc = MetadataEncoder({"channel_type": ["news", "sports"]})
    vec = build_feature_vector({"channel_type": "news"}, "goal scored", enc, HashingEmbedder(n_features=8))
    assert vec.shape == (3 + 8,)
    assert vec[:3].tolist() == [1.0, 0.0, 0.0]


def test_build_feature_vector_without_an_image_embedder_ignores_image_bytes() -> None:
    enc = MetadataEncoder({"channel_type": ["news"]})
    with_bytes = build_feature_vector({"channel_type": "news"}, "x", enc, HashingEmbedder(n_features=8), b"abc")
    without_bytes = build_feature_vector({"channel_type": "news"}, "x", enc, HashingEmbedder(n_features=8))
    assert with_bytes.shape == without_bytes.shape == (2 + 8,)


def test_build_feature_vector_with_an_image_appends_the_embedding_and_a_present_flag() -> None:
    enc = MetadataEncoder({"channel_type": ["news"]})
    image_embedder = FakeImageEmbedder()
    vec = build_feature_vector(
        {"channel_type": "news"}, "x", enc, HashingEmbedder(n_features=8), b"abcd", image_embedder
    )
    assert vec.shape == (2 + 8 + 4 + 1,)
    assert vec[-5:].tolist() == [4.0, 1.0, 2.0, 3.0, 1.0]  # len(b"abcd")=4, then the fake's vector, then flag=1
    assert image_embedder.calls == [b"abcd"]


def test_build_feature_vector_with_an_embedder_but_no_image_for_this_example() -> None:
    enc = MetadataEncoder({"channel_type": ["news"]})
    image_embedder = FakeImageEmbedder()
    vec = build_feature_vector({"channel_type": "news"}, "x", enc, HashingEmbedder(n_features=8), None, image_embedder)
    assert vec.shape == (2 + 8 + 4 + 1,)
    assert vec[-5:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]  # zero image vector, flag=0
    assert image_embedder.calls == []  # never called for an example with no image
