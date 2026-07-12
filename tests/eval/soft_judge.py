"""Non-cyclic soft-intent judging over objective song attributes.

The judge deliberately avoids generated explanations. It only reads fields that
come from retrieval metadata or deterministic enrichment, so it can be calibrated
against a human gold set without becoming a self-approval loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SoftJudgeDecision:
    status: str
    detail: str
    confidence: float
    metrics: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _field_values(song: dict[str, Any], *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        val = song.get(name)
        if name in {"instrumental", "is_instrumental"} and isinstance(val, bool):
            values.append("instrumental" if val else "vocal")
        elif isinstance(val, list):
            values.extend(norm(x) for x in val if norm(x))
        elif norm(val):
            values.append(norm(val))
    return values


def objective_tokens(song: dict[str, Any]) -> list[str]:
    """Return tokens from objective metadata only.

    Deliberately excludes title, artist, recommendation reason, and final
    explanation. This keeps the judge from validating text the system wrote
    about itself.
    """
    tokens: list[str] = []
    for value in _field_values(
        song,
        "genre",
        "genres",
        "moods",
        "scenarios",
        "language",
        "region",
        "instrumental",
        "is_instrumental",
    ):
        normalized = norm(value)
        if normalized:
            tokens.append(normalized)
        for part in normalized.replace("/", " ").replace("|", " ").replace(",", " ").replace("，", " ").split():
            if part:
                tokens.append(part)
    return sorted(set(tokens))


_TAG_SYNONYMS: dict[str, set[str]] = {
    "relaxing": {"peaceful", "calm", "soothing", "healing", "chill", "soft", "gentle"},
    "warm": {"cozy", "comforting", "healing", "gentle", "acoustic"},
    "mellow": {"soft", "gentle", "calm", "laid-back", "lo-fi", "acoustic"},
    "happy": {"cheerful", "joyful", "pleasant", "bright"},
}


def _wanted_terms(item: str) -> list[str]:
    wanted_norm = norm(item)
    if not wanted_norm:
        return []
    expanded = [wanted_norm]
    expanded.extend(sorted(_TAG_SYNONYMS.get(wanted_norm, set())))
    return expanded


def tag_hit(tokens: list[str], wanted: list[str]) -> bool:
    for item in wanted:
        for wanted_norm in _wanted_terms(item):
            for token in tokens:
                token_norm = norm(token)
                if wanted_norm == token_norm:
                    return True
                # Keep broad partial matches for descriptive phrases, but do not
                # let short parent tags satisfy specific labels: "pop" is not
                # "k-pop", and "r&b" is not a longer neighboring tag.
                if token_norm in wanted_norm and len(token_norm) < 4:
                    continue
                if wanted_norm in token_norm or token_norm in wanted_norm:
                    return True
    return False


def _confidence(
    passed: bool,
    coverage_ratio: float,
    pos_ratio: float,
    neg_ratio: float,
    min_coverage: float,
    min_positive: float,
    max_negative: float,
) -> float:
    """Coarse confidence for reporting, not for pass/fail decisions."""
    if passed:
        margin = min(
            coverage_ratio - min_coverage,
            pos_ratio - min_positive,
            max_negative - neg_ratio,
        )
    else:
        margin = max(
            min_coverage - coverage_ratio,
            min_positive - pos_ratio,
            neg_ratio - max_negative,
        )
    return max(0.0, min(1.0, 0.5 + margin))


def judge_objective_soft_intent(songs: list[dict[str, Any]], config: dict[str, Any]) -> SoftJudgeDecision:
    """Judge soft intent with configured positive/negative objective tags.

    Config keys:
    - positive_any: tags where at least one may match a song.
    - negative_any: tags that should be rare or absent.
    - min_positive_ratio: minimum positive tag hit ratio among covered songs.
    - max_negative_ratio: maximum negative tag hit ratio among covered songs.
    - min_coverage_ratio: minimum fraction of songs with objective tag metadata.
    """
    if not isinstance(config, dict):
        return SoftJudgeDecision("skip", "objective_soft_judge 配置不是对象，跳过", 0.0)
    if not songs:
        return SoftJudgeDecision("fail", "无结果", 1.0)

    positive = [norm(x) for x in config.get("positive_any", [])]
    negative = [norm(x) for x in config.get("negative_any", [])]
    min_positive = float(config.get("min_positive_ratio", 0.0))
    max_negative = float(config.get("max_negative_ratio", 1.0))
    min_coverage = float(config.get("min_coverage_ratio", 0.5))
    if not positive and not negative:
        return SoftJudgeDecision("skip", "未提供 positive_any/negative_any，跳过", 0.0)

    tokenized = [(song, objective_tokens(song)) for song in songs]
    covered = [(song, tokens) for song, tokens in tokenized if tokens]
    if not covered:
        return SoftJudgeDecision("skip", "返回歌曲缺少可判定的客观标签字段", 0.0)

    coverage_ratio = len(covered) / len(songs)
    pos_hits = sum(1 for _, tokens in covered if tag_hit(tokens, positive)) if positive else len(covered)
    neg_hits = sum(1 for _, tokens in covered if tag_hit(tokens, negative)) if negative else 0
    pos_ratio = pos_hits / len(covered)
    neg_ratio = neg_hits / len(covered)

    failures = []
    if coverage_ratio < min_coverage:
        failures.append(f"coverage {len(covered)}/{len(songs)} = {coverage_ratio:.0%}，要求 ≥ {min_coverage:.0%}")
    if positive and pos_ratio < min_positive:
        failures.append(f"positive {pos_hits}/{len(covered)} = {pos_ratio:.0%}，要求 ≥ {min_positive:.0%}")
    if negative and neg_ratio > max_negative:
        failures.append(f"negative {neg_hits}/{len(covered)} = {neg_ratio:.0%}，要求 ≤ {max_negative:.0%}")

    metrics = {
        "coverage_ratio": coverage_ratio,
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "covered_count": float(len(covered)),
        "song_count": float(len(songs)),
    }
    evidence = {
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "sample_tokens": [tokens for _, tokens in covered[:3]],
    }
    passed = not failures
    confidence = _confidence(
        passed=passed,
        coverage_ratio=coverage_ratio,
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        min_coverage=min_coverage,
        min_positive=min_positive,
        max_negative=max_negative,
    )
    if failures:
        return SoftJudgeDecision("fail", "; ".join(failures), confidence, metrics, evidence)

    detail = (
        f"coverage {len(covered)}/{len(songs)} = {coverage_ratio:.0%}; "
        f"positive {pos_hits}/{len(covered)} = {pos_ratio:.0%}; "
        f"negative {neg_hits}/{len(covered)} = {neg_ratio:.0%}; "
        f"confidence={confidence:.2f}"
    )
    return SoftJudgeDecision("pass", detail, confidence, metrics, evidence)
