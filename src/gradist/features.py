"""Feature vector: metadata one-hot (with an unknown bucket) + text embedding + an image embedding.

Nothing here knows about any specific company's metadata values. A consumer declares its own
known values per field; anything else falls into that field's "unknown" column rather than
raising or being silently dropped.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import HashingVectorizer

Vector = NDArray[np.float64]


def load_image_bytes(path: str | Path) -> bytes:
    """Read an image file's raw bytes, for a `TierLabel.image_path` at training time. Over the
    service API an upload arrives as bytes already and skips this."""
    return Path(path).read_bytes()


class Embedder(Protocol):
    """Same shape as the text-embedding interface used elsewhere, so a real model drops in later."""

    def embed(self, texts: list[str]) -> Vector: ...


class HashingEmbedder:
    """Character n-gram hashing. Lexical, not semantic — for tests and as a floor, not for
    production routing quality.
    """

    def __init__(self, n_features: int = 4096, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            analyzer="char_wb",
            ngram_range=ngram_range,
            alternate_sign=False,
            norm="l2",
        )

    def embed(self, texts: list[str]) -> Vector:
        return np.asarray(self._vectorizer.transform(texts).toarray(), dtype=np.float64)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model, for example BAAI/bge-m3 (multilingual, MIT — see the
    earlier project's ADR-0012 for why). Needs the `embeddings` extra.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install .[embeddings]"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> Vector:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float64)


class MetadataEncoder:
    """One-hot encoder over caller-declared fields, each with its own "unknown" column.

    `fields` is `{field_name: [known_value, ...]}`. A value present for a declared field that is
    not in its known list, or a declared field missing from the input, encodes to that field's
    unknown column (never an error, never silently zero everywhere). A field name NOT declared
    at construction time is a caller mistake and raises, so a typo cannot be lost in a feature
    vector that quietly has one less signal than intended.
    """

    def __init__(self, fields: dict[str, list[str]]) -> None:
        if not fields:
            raise ValueError("MetadataEncoder needs at least one field")
        self._columns: dict[str, list[str]] = {}
        for name, values in fields.items():
            deduped = list(dict.fromkeys(values))  # keep order, drop duplicates
            self._columns[name] = [*deduped, "__unknown__"]

    @property
    def width(self) -> int:
        return sum(len(cols) for cols in self._columns.values())

    @property
    def field_names(self) -> list[str]:
        return [name for name in self._columns if name != "_unused"]

    def known_only(self, metadata: dict[str, str]) -> dict[str, str]:
        """`metadata` without undeclared fields — for live requests, which may carry anything."""
        return {k: v for k, v in metadata.items() if k in self._columns}

    def bucket_label(self, field: str, value: str | None) -> str:
        """Which column `value` falls into for `field`: the value itself, or "__unknown__"."""
        if field not in self._columns:
            raise KeyError(f"undeclared metadata field '{field}'. Declared: {sorted(self._columns)}")
        known = self._columns[field][:-1]
        return value if value in known else "__unknown__"

    def encode(self, metadata: dict[str, str]) -> Vector:
        unknown_keys = set(metadata) - set(self._columns)
        if unknown_keys:
            raise KeyError(
                f"undeclared metadata field(s): {sorted(unknown_keys)}. "
                f"Declared fields: {sorted(self._columns)}"
            )
        parts: list[Vector] = []
        for name, values in self._columns.items():
            row = np.zeros(len(values), dtype=np.float64)
            bucket = self.bucket_label(name, metadata.get(name))
            row[values.index(bucket)] = 1.0
            parts.append(row)
        return np.concatenate(parts)


class ImageEmbedder(Protocol):
    """An image embedder takes raw image bytes (PNG/JPEG/etc.) and returns one vector per image."""

    width: int

    def embed(self, images: list[bytes]) -> Vector: ...


class Dinov2Embedder:
    """Real image embedding via facebook/dinov2-small.

    License checked 2026-11-16: that model card carries an explicit `license: apache-2.0` tag.
    sentence-transformers/clip-ViT-B-32 was checked too and rejected — its card has NO license
    tag at all, the same ambiguous-license exclusion this project already applied to Arch-Router
    (the earlier R&D project's ADR-0006). DINOv2 is a pure vision embedding, not a joint
    image-text space like CLIP — fine here, since only an image feature is needed, not
    text-image alignment. Needs the `images` extra (transformers, torch, pillow).
    """

    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers/torch/pillow are not installed. Install with: pip install .[images]"
            ) from exc
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self.width = int(self._model.config.hidden_size)

    def embed(self, images: list[bytes]) -> Vector:
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]  # the pooled [CLS] embedding, per image
        return np.asarray(cls_token.numpy(), dtype=np.float64)


def build_feature_vector(
    metadata: dict[str, str],
    text: str,
    metadata_encoder: MetadataEncoder,
    text_embedder: Embedder,
    image_bytes: bytes | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> Vector:
    """Metadata one-hot, concatenated with the text embedding, concatenated with the image
    embedding (when `image_embedder` is configured at all — training and prediction must agree
    on this, or the feature vector width would differ between examples).

    When `image_embedder` is given but THIS example has no image, a zero vector of the image
    embedder's width is used instead, plus one extra flag column (1.0 with an image, 0.0
    without) — so "no image" is a visible, distinct signal, never confused with an image that
    happened to embed near zero.
    """
    meta_vec = metadata_encoder.encode(metadata)
    text_vec = text_embedder.embed([text])[0]
    parts = [meta_vec, text_vec]
    if image_embedder is not None:
        if image_bytes is not None:
            parts.append(image_embedder.embed([image_bytes])[0])
            parts.append(np.array([1.0]))
        else:
            parts.append(np.zeros(image_embedder.width, dtype=np.float64))
            parts.append(np.array([0.0]))
    return np.concatenate(parts)
