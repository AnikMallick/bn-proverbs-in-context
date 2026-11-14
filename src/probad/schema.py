from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, ValidationError

from probad.utils import content_words, normalise_text, read_jsonl, words, write_jsonl, load_english_stopwords, load_bengali_stopwords

EN_STOPWORDS = load_english_stopwords()

#### dataclasses ####


class Turn(BaseModel):
    speaker: str = Field(..., description="speaker in a conversation")
    text: str = Field(..., description="spoken text in a conversation by the speaker")


class Options(BaseModel):
    correct: str = Field(..., description="correct meaning of the proverb")
    literal: str = Field(..., description="literal meaning of the proverb")
    plausible: str = Field(..., description="plausible meaning of the proverb")

    def get(self, key: str) -> str:
        return getattr(self, key)


class LangBlock(BaseModel):
    """Everything language-specific: conversation, proverb speaker, options."""

    turns: list[Turn] = Field(..., description="Conversation turns")
    proverb_speaker: str = Field(..., description="speaker of the proverb")
    options: Options = Field(..., description="meanings of the proverb")

    def speaker_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == self.proverb_speaker]

    def context_text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)


class Meaning(BaseModel):
    en: str = Field(..., description="meanings in english")
    bn: str | None = Field(None, description="meanings in bengal")


class Proverb(BaseModel):
    proverb_id: str = Field(..., description="proverb id")
    bengali: str = Field(..., description="proverb in bengali characters")
    romanised: str = Field(..., description="proverb in romanized characters")
    literal_gloss: str = Field(..., description="literal meaning in english")
    meaning: Meaning = Field(..., description="meanings in different languages")
    figurative: bool = Field(..., description="proverb is figurative")
    region: Literal["both", "wb", "bd"]  = Field("both", description="proverb region: both, wb, bd")
    cultural_note: str = Field("", description="cultural reference")
    false_friends: list[str] = Field([], description="proverb token to eng tokens which are similar") # e.g. ["kola→cola"]
    respelled: str | None = Field(None, description="proverb respelled romanized")

    @property
    def has_false_friends(self) -> bool:
        return bool(self.false_friends)


class MemorisationProbe(BaseModel):
    prefix: str = Field(..., description="memorization prefix")
    target: str = Field(..., description="memorization target")


class Memorisation(BaseModel):
    bengali: MemorisationProbe | None = Field(None, description="memorization in bengali")
    romanised: MemorisationProbe | None = Field(None, description="memorization in romanized")


class MapsLink(BaseModel):
    in_maps: bool = Field(False, description="is in MAPS")
    maps_ref: str | None = Field(None, description="if yes then MAPS ref")


class Annotator2(BaseModel):
    label: str | None = Field(None, description="second annotator label")
    agreed: bool | None = Field(None, description="second annotator agreed")
    comment: str | None = Field(None, description="second annotator comment")


class Meta(BaseModel):
    split: Literal["train", "test"] = Field("test", description="data split")
    subsets: list[str] = Field([], description="subset")
    author: str = Field("", description="author of this entry")
    created: str = Field("", description="created at")
    updated: str = Field("", description="updated at")
    romanisation_scheme: str = Field("v1")
    stereotype_flag: bool = Field(False, description="")
    annotator2: Annotator2 = Field(default_factory=Annotator2, description="")
    source: str | None = Field(None)  # e.g. "human", "synthetic:<model>"
    
    @property
    def is_human(self) -> bool:
        return self.source == "human"

    @property
    def which_model(self) -> str | None:
        if self.is_human: return None
        if self.source.startswith("synthetic:"):
            return "".join(self.source.split(":")[1:])


class TestItem(BaseModel):
    """One ProBaD item. Used for the human test set and for synthetic training data."""

    item_id: str = Field(..., description="data item id")
    proverb: Proverb = Field(..., description="proverb data")
    en: LangBlock = Field(..., description="english conversation")
    bn: LangBlock | None = Field(None, description="bengali conversation")
    memorisation: Memorisation = Field(default_factory=Memorisation, description="memorisation details")
    maps: MapsLink = Field(default_factory=MapsLink, description="MAPS ref")
    meta: Meta = Field(default_factory=Meta, description="metadata")
    version: int = Field(1, description="version")

    # helpers
    def block(self, lang: str) -> LangBlock | None:
        if lang == "en":
            return self.en
        if lang == "bn":
            return self.bn
        raise ValueError(f"unknown lang {lang!r}")

    def has_lang(self, lang: str) -> bool:
        return self.block(lang) is not None

    def meaning(self, lang: str) -> str | None:
        return self.proverb.meaning.en if lang == "en" else self.proverb.meaning.bn

    # loaders
    @staticmethod
    def load_items(path: str | Path) -> Iterator[TestItem]:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield TestItem.model_validate_json(line)
                except ValidationError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})")
    
    @staticmethod
    def load_item(path: str | Path) -> TestItem:
        with open(path, encoding="utf-8") as f:
            return TestItem.model_validate_json(f.read())

    def validate_item(self, BN_STOPWORDS:list[str], *, strict_bn_overlap: float = 0.5) -> list[Issue]:
        """Validate one item (dict or TestItem). Returns a list of issues (empty = clean)."""
        issues: list[Issue] = []
        p = self.proverb
        item_id = self.item_id
        # ---- per-language block checks
        for lang in ("en", "bn"):
            block = self.block(lang)
            if block is None:
                continue
            speakers = {t.speaker for t in block.turns}
            if block.proverb_speaker not in speakers:
                issues.append(Issue(item_id=item_id, level="error", 
                                    code=f"{lang}.speaker", 
                                    message=f"proverb_speaker {block.proverb_speaker!r} not in turns"))
                continue
            phrase = p.romanised if lang == "en" else p.bengali
            if not any(find_span(t.text, phrase) for t in block.speaker_turns()):
                issues.append(
                    Issue(
                        item_id = item_id,
                        level="error",
                        code=f"{lang}.proverb_missing",
                        message=f"proverb text not found in a turn of {block.proverb_speaker!r} ({phrase!r})",
                    )
                )
            opts = [block.options.correct, block.options.literal, block.options.plausible]
            if len({normalise_text(o) for o in opts}) < 3:
                issues.append(Issue(item_id=item_id, level="error", code=f"{lang}.options_duplicate", message="two options are identical"))
            ctx = normalise_text(block.context_text())
            if normalise_text(block.options.correct) in ctx:
                issues.append(Issue(item_id=item_id, level="error", code=f"{lang}.answer_in_context", message="correct option appears verbatim in context"))
            lens = [len(words(o)) for o in opts]
            if max(lens) > 3 * max(1, min(lens)):
                issues.append(Issue(item_id=item_id, level="warning", code=f"{lang}.length_imbalance", message=f"option word counts very uneven: {lens}"))
            if len(block.turns) > 4:
                issues.append(Issue(item_id=item_id, level="warning", code=f"{lang}.long_context", message=f"{len(block.turns)} turns (guideline 1-3)"))

        # ---- Bengali literal distractor must paraphrase, not copy the proverb words
        if self.bn is not None:
            n_shared, frac = _word_overlap(self.bn.options.literal, p.bengali, BN_STOPWORDS)
            if frac > strict_bn_overlap:
                issues.append(
                    Issue(
                        item_id=item_id,
                        level="warning",
                        code="bn.literal_copies_proverb",
                        message=f"Bengali literal distractor shares {n_shared} content words ({frac:.0%}) with the proverb; paraphrase where possible",
                    )
                )
            if not p.meaning.bn:
                issues.append(Issue(item_id=item_id, level="warning", code="bn.meaning_missing", message="bn block present but meaning.bn empty"))

        # ---- English literal distractor should not be the gloss verbatim (gives a copy cue in gloss render)
        if normalise_text(self.en.options.literal) == normalise_text(p.literal_gloss):
            issues.append(
                Issue(item_id=item_id, level="warning", code="en.literal_equals_gloss", message="literal option identical to literal_gloss (copy cue in gloss render)")
            )
        _, frac_en = _word_overlap(self.en.options.correct, p.literal_gloss, EN_STOPWORDS)
        if frac_en > 0.5:
            issues.append(Issue(item_id=item_id, level="warning", code="en.correct_echoes_gloss", message="correct option reuses many literal-gloss words"))

        # ---- false friends / respelling
        if p.false_friends and not p.respelled:
            issues.append(Issue(item_id=item_id, level="error", code="respelled_missing", message="false_friends set but respelled is empty"))
        if p.respelled and normalise_text(p.respelled) == normalise_text(p.romanised):
            issues.append(Issue(item_id=item_id, level="error", code="respelled_same", message="respelled equals romanised"))
        if p.respelled and len(words(p.respelled)) != len(words(p.romanised)):
            issues.append(Issue(item_id=item_id, level="warning", code="respelled_wordcount", message="respelling should change words, not add/remove them"))

        # ---- memorisation probes
        for script, probe, full in (
            ("bengali", self.memorisation.bengali, p.bengali),
            ("romanised", self.memorisation.romanised, p.romanised),
        ):
            if probe is None:
                issues.append(Issue(item_id=item_id, level="warning", code=f"memorisation.{script}_missing", message="no probe"))
                continue
            joined_w = words(f"{probe.prefix} {probe.target}")
            full_w = words(full)
            if full_w[: len(joined_w)] != joined_w:
                issues.append(
                    Issue(
                        item_id=item_id,
                        level="warning",
                        code=f"memorisation.{script}_mismatch",
                        message="prefix + target does not reproduce the proverb",
                    )
                )

        # ---- subsets / maps / meta
        if "native_vs_translated" in self.meta.subsets:
            if self.bn is None:
                issues.append(Issue(item_id=item_id, level="error", code="nvt.bn_missing", message="native_vs_translated item needs a bn block"))
            if not self.maps.in_maps or not self.maps.maps_ref:
                issues.append(Issue(item_id=item_id, level="error", code="nvt.maps_ref", message="native_vs_translated item needs maps.in_maps and maps_ref"))
        if self.meta.stereotype_flag and self.meta.split == "test":
            issues.append(Issue(item_id=item_id, level="warning", code="stereotype", message="stereotype-flagged proverb in the test split"))
        a2 = self.meta.annotator2
        if a2.label is not None and a2.agreed is None:
            issues.append(Issue(item_id=item_id, level="warning", code="annotator2.unadjudicated", message="annotator label present but not adjudicated"))
        return issues

class Issue(BaseModel):
    item_id: str = Field(...)
    level: Literal["error", "warning"] = Field(...)  # "error" | "warning"
    code: str  = Field(...)
    message: str = Field(...)

    def __str__(self) -> str:
            return f"[{self.level.upper()}] {self.item_id}: {self.code}: {self.message}"

#### functions ####

def find_span(text: str, phrase: str) -> tuple[int, int] | None:
    """Find ``phrase`` inside ``text`` tolerantly (case, punctuation, spacing).

    Returns (start, end) character offsets into ``text`` or None.
    """
    import re

    if not phrase:
        return None
    idx = text.find(phrase)
    if idx >= 0:
        return idx, idx + len(phrase)
    core = phrase.strip().rstrip(".!?।").strip()
    idx = text.find(core)
    if idx >= 0:
        return idx, idx + len(core)
    toks = [re.escape(t) for t in words(core, lower=False)]
    if not toks:
        return None
    pattern = r"[\s\W]*".join(toks)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return None


def _word_overlap(a: str, b: str, stopwords: frozenset[str]) -> tuple[int, float]:
    wa = set(content_words(a, stopwords))
    wb = set(content_words(b, stopwords))
    if not wb:
        return 0, 0.0
    shared = wa & wb
    return len(shared), len(shared) / len(wb)

def validate_items(items: list[TestItem],BN_STOPWORDS: list[str],  *, strict_bn_overlap: float = 0.5) -> list[Issue]:
    issues: list[Issue] = []
    for item in items: issues.extend(item.validate_item(BN_STOPWORDS, strict_bn_overlap=strict_bn_overlap))
    return issues