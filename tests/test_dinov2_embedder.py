"""Real image-model test: downloads facebook/dinov2-small the first time, needs the `images`
extra. Skipped by default so the normal fast test suite never needs torch or a network call.

Run explicitly: GRADIST_TEST_REAL_IMAGE_MODEL=1 pytest tests/test_dinov2_embedder.py
"""

from __future__ import annotations

import io
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRADIST_TEST_REAL_IMAGE_MODEL"),
    reason="set GRADIST_TEST_REAL_IMAGE_MODEL=1 to download and run the real image model",
)


def _solid_color_png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_dinov2_embedder_loads_and_embeds_a_real_image() -> None:
    from gradist.features import Dinov2Embedder

    embedder = Dinov2Embedder()
    assert embedder.width > 0
    red = embedder.embed([_solid_color_png((220, 20, 20))])
    blue = embedder.embed([_solid_color_png((20, 20, 220))])
    assert red.shape == (1, embedder.width)
    # two very different images must not embed identically
    assert not (red == blue).all()


def test_dinov2_embedder_is_deterministic_for_the_same_image() -> None:
    from gradist.features import Dinov2Embedder

    embedder = Dinov2Embedder()
    image = _solid_color_png((10, 200, 10))
    a = embedder.embed([image])
    b = embedder.embed([image])
    assert (a == b).all()
