from __future__ import annotations

from dataclasses import dataclass


class SimilarityEvaluator:
    """Decides whether two items should be labeled as 'correct'/similar."""

    def answers_similar(self, a: str, b: str, *, id_set_a: int | None = None, id_set_b: int | None = None) -> bool:  # noqa: D401
        raise NotImplementedError


@dataclass(frozen=True)
class IdSetSimilarityEvaluator(SimilarityEvaluator):
    """Label is 1 iff id_set_a == id_set_b and both are usable."""

    unusable_value: int = -1

    def answers_similar(self, a: str, b: str, *, id_set_a: int | None = None, id_set_b: int | None = None) -> bool:
        if id_set_a is None or id_set_b is None:
            return False
        try:
            ia = int(id_set_a)
            ib = int(id_set_b)
        except Exception:
            return False
        if ia == self.unusable_value or ib == self.unusable_value:
            return False
        return ia == ib


def _norm_answer(s: str) -> str:
    return (
        str(s)
        .strip()
        .rstrip(".")
        .lower()
        .replace('"', "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
    )


@dataclass(frozen=True)
class StringComparisonSimilarityEvaluator(SimilarityEvaluator):
    """Label is 1 iff normalized strings are exactly equal."""

    def answers_similar(self, a: str, b: str, *, id_set_a: int | None = None, id_set_b: int | None = None) -> bool:
        return _norm_answer(a) == _norm_answer(b)


def choose_id_set_column(columns: list[str]) -> str | None:
    if "ID_Set" in columns:
        return "ID_Set"
    if "id_set" in columns:
        return "id_set"
    return None


def has_usable_ids(values) -> bool:
    """Best-effort check for presence of any usable integer IDs."""
    try:
        for v in values:
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            if iv != -1:
                return True
        return False
    except Exception:
        return False

