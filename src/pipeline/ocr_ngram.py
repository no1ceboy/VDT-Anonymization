"""Corpus-trained, conservative OCR spacing normalization for Vietnamese text.

The normalizer only removes whitespace between alphabetic fragments when the
joined token occurs in the training corpus and a character-trigram model
prefers the joined local context. It also normalizes Unicode spacing while
keeping a mapping back to the untouched source offsets.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_FORMAT = "vdt-character-trigram-spacing-v1"
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
START = "\u0002"
END = "\u0003"


def _language_text(text: str) -> str:
    """Canonicalize text for language-model counts without changing letters."""
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(text.split())


@dataclass
class OffsetMappedText:
    text: str
    source_starts: array
    source_ends: array
    whitespace_changes: int = 0
    joins: list[dict] | None = None

    def source_span(self, start: int, end: int) -> tuple[int, int] | None:
        """Map a nonempty normalized span to a covering source span."""
        if start < 0 or end <= start or end > len(self.text):
            return None
        return int(self.source_starts[start]), int(self.source_ends[end - 1])


class CharacterTrigramModel:
    def __init__(self, grams: dict[str, int], contexts: dict[str, int],
                 candidate_words: dict[str, int], alphabet_size: int,
                 metadata: dict | None = None):
        self.grams = grams
        self.contexts = contexts
        self.candidate_words = candidate_words
        self.alphabet_size = max(16, int(alphabet_size))
        self.metadata = metadata or {}

    @classmethod
    def train(cls, texts: Iterable[str], min_word_count: int = 3,
              progress=None) -> "CharacterTrigramModel":
        grams: dict[str, int] = {}
        contexts: dict[str, int] = {}
        words: dict[str, int] = {}
        alphabet: set[str] = {START, END, " "}
        documents = 0
        characters = 0
        token_total = 0

        for raw_text in texts:
            if not raw_text:
                continue
            text = _language_text(raw_text)
            if not text:
                continue
            documents += 1
            characters += len(text)
            alphabet.update(text)
            for match in TOKEN_RE.finditer(text):
                token = match.group().casefold()
                words[token] = words.get(token, 0) + 1
                token_total += 1

            padded = START + START + text + END
            for index in range(len(padded) - 2):
                gram = padded[index:index + 3]
                context = gram[:2]
                grams[gram] = grams.get(gram, 0) + 1
                contexts[context] = contexts.get(context, 0) + 1

            if progress and documents % 500 == 0:
                progress(documents, characters, token_total)

        candidate_words = {
            word: count for word, count in words.items()
            if count >= min_word_count and len(word) <= 24
        }
        metadata = {
            "documents": documents,
            "characters": characters,
            "tokens": token_total,
            "candidate_words": len(candidate_words),
            "min_word_count": min_word_count,
        }
        return cls(grams, contexts, candidate_words, len(alphabet), metadata)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT,
            "alphabet_size": self.alphabet_size,
            "metadata": self.metadata,
            "grams": self.grams,
            "contexts": self.contexts,
            "candidate_words": self.candidate_words,
        }
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTrigramModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported OCR n-gram model format: {payload.get('format')!r}")
        return cls(
            grams={str(key): int(value) for key, value in payload["grams"].items()},
            contexts={str(key): int(value) for key, value in payload["contexts"].items()},
            candidate_words={str(key): int(value) for key, value in payload["candidate_words"].items()},
            alphabet_size=int(payload.get("alphabet_size", 128)),
            metadata=payload.get("metadata", {}),
        )

    def mean_log_probability(self, text: str) -> float:
        text = _language_text(text)
        padded = START + START + text + END
        total = 0.0
        count = max(1, len(padded) - 2)
        smoothing = 0.1
        for index in range(count):
            gram = padded[index:index + 3]
            context = gram[:2]
            probability = (
                self.grams.get(gram, 0) + smoothing
            ) / (
                self.contexts.get(context, 0) + smoothing * self.alphabet_size
            )
            total += math.log(probability)
        return total / count


def _normalize_unicode_spacing(text: str) -> OffsetMappedText:
    output: list[str] = []
    starts = array("I")
    ends = array("I")
    changes = 0
    index = 0

    while index < len(text):
        char = text[index]
        if char in ZERO_WIDTH:
            changes += 1
            index += 1
            continue

        source_end = index + 1
        if char == "\r":
            if source_end < len(text) and text[source_end] == "\n":
                source_end += 1
            normalized = "\n"
            if char != normalized or source_end > index + 1:
                changes += 1
        elif char == "\n":
            normalized = "\n"
        elif char.isspace():
            normalized = " "
            if char != " ":
                changes += 1
        else:
            normalized = char

        if normalized == " " and output and output[-1] == " ":
            ends[-1] = source_end
            changes += 1
        else:
            output.append(normalized)
            starts.append(index)
            ends.append(source_end)
        index = source_end

    return OffsetMappedText("".join(output), starts, ends, changes, [])


def normalize_before_ner(text: str, model: CharacterTrigramModel,
                         min_gain: float = 0.18, min_candidate_frequency: int = 3,
                         context_chars: int = 8, max_fragments: int = 4,
                         max_candidate_chars: int = 24) -> OffsetMappedText:
    """Normalize spacing and apply high-confidence corpus-supported joins.

    Newlines and punctuation block joins. Candidate groups are selected
    greedily by n-gram gain, and no source character is rewritten or lost
    outside whitespace. The returned offsets still refer to ``text``.
    """
    mapped = _normalize_unicode_spacing(text)
    base_text = mapped.text
    tokens = list(re.finditer(r"\S+", base_text))
    candidates: list[dict] = []

    for start_index, first in enumerate(tokens):
        if not first.group().isalpha():
            continue
        if len(first.group()) == 1 and first.group().isascii() and first.group().isupper():
            # Preserve visible anonymization initials such as A, H, and K.
            continue
        joined_parts = [first.group()]
        previous = first
        for end_index in range(start_index + 1, min(len(tokens), start_index + max_fragments)):
            current = tokens[end_index]
            separator = base_text[previous.end():current.start()]
            if separator != " " or not current.group().isalpha():
                break
            if len(current.group()) == 1 and current.group().isascii() and current.group().isupper():
                break
            joined_parts.append(current.group())
            previous = current
            joined = "".join(joined_parts)
            if len(joined) > max_candidate_chars:
                break

            key = unicodedata.normalize("NFC", joined).casefold()
            if model.candidate_words.get(key, 0) < min_candidate_frequency:
                continue

            local_start = max(0, first.start() - context_chars)
            local_end = min(len(base_text), current.end() + context_chars)
            raw_context = base_text[local_start:local_end]
            candidate_context = (
                base_text[local_start:first.start()]
                + joined
                + base_text[current.end():local_end]
            )
            gain = (
                model.mean_log_probability(candidate_context)
                - model.mean_log_probability(raw_context)
            )
            if gain >= min_gain:
                candidates.append({
                    "token_start": start_index,
                    "token_end": end_index,
                    "start": first.start(),
                    "end": current.end(),
                    "gain": gain,
                    "parts": len(joined_parts),
                })

    # Resolve overlapping candidate groups deterministically, preferring the
    # strongest n-gram evidence and then the larger repaired fragment.
    selected: list[dict] = []
    occupied: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: (-item["gain"], -item["parts"], item["start"])):
        span = set(range(candidate["token_start"], candidate["token_end"] + 1))
        if occupied.isdisjoint(span):
            selected.append(candidate)
            occupied.update(span)

    remove_positions: set[int] = set()
    for candidate in selected:
        group = tokens[candidate["token_start"]:candidate["token_end"] + 1]
        for left, right in zip(group, group[1:]):
            remove_positions.update(range(left.end(), right.start()))

    final_chars: list[str] = []
    final_starts = array("I")
    final_ends = array("I")
    for index, char in enumerate(base_text):
        if index in remove_positions:
            continue
        final_chars.append(char)
        final_starts.append(mapped.source_starts[index])
        final_ends.append(mapped.source_ends[index])

    join_records = []
    for candidate in sorted(selected, key=lambda item: item["start"]):
        source_span = mapped.source_span(candidate["start"], candidate["end"])
        if source_span:
            join_records.append({
                "source_start": source_span[0],
                "source_end": source_span[1],
                "parts": candidate["parts"],
                "ngram_gain": round(candidate["gain"], 5),
            })

    return OffsetMappedText(
        text="".join(final_chars),
        source_starts=final_starts,
        source_ends=final_ends,
        whitespace_changes=mapped.whitespace_changes,
        joins=join_records,
    )
