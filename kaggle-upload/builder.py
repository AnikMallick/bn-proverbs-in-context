"""Turn stored items into scorable prompts.

For each item the builder generates every requested combination of

    language (en / bn) x render x format x control x option order

as a ``PromptInstance``. Each instance carries two prompt strings:

* ``prompt``       : lettered multiple-choice prompt ending with ``Answer:``
                     (used by letter logprob scoring), and
* ``cloze_prompt`` : the same without the option list (used by option-text
                     scoring, where each option text is scored as a continuation).

The builder never touches a model; system prompts / chat templates are applied
at scoring time, so the same instances serve every condition.
"""

from __future__ import annotations

from functools import cached_property
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from probad.constants import (
    CONTROLS,
    FIXED_TEXT,
    FORMAT_GOLD,
    FORMAT_OPTIONS,
    FORMATS,
    LANGS,
    LETTERS,
    NONE_KEY,
    RENDERS,
)
from probad.schema import TestItem, find_span

_END_PUNCT = ".!?।"


class BuildError(ValueError):
    pass


class PromptInstance(BaseModel):
    instance_id: str = Field(...)
    item_id: str = Field(...)
    proverb_id: str = Field(...)
    lang: str = Field(...)
    render: str = Field(...)
    format: str = Field(...)
    control: str = Field(...)
    option_keys: list[str] = Field(...)  # option keys in display order, e.g. ["literal", "correct"]
    option_texts: list[str] = Field(...)  # texts in the same order
    gold_key: str = Field(...)
    prompt: str = Field(...)
    cloze_prompt: str = Field(...)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def letters(self) -> list[str]:
        return list(LETTERS[: len(self.option_keys)])

    @property
    def gold_letter(self) -> str:
        return self.letters[self.option_keys.index(self.gold_key)]

    def key_of(self, letter: str) -> str:
        return self.option_keys[self.letters.index(letter)]

    def text_of(self, key: str) -> str:
        return self.option_texts[self.option_keys.index(key)]

    # def to_dict(self) -> dict[str, Any]:
    #     return asdict(self)

    # @classmethod
    # def from_dict(cls, d: dict[str, Any]) -> PromptInstance:
    #     return cls(**d)


@dataclass
class BuildConfig:
    formats: Sequence[str] = ("normal", "negative", "none", "meaning")
    renders: Sequence[str] = ("romanised",)
    langs: Sequence[str] = ("en",)
    controls: Sequence[str] = ("none",)
    all_orders: bool = True  # every permutation of the option order
    none_last: bool = True  # keep "None of these" as the last option in the `none` format
    instruction: str | None = None  # optional line placed before the prompt
    skip_unavailable: bool = True  # silently skip renders/langs an item lacks (else raise)

    def __post_init__(self) -> None:
        for name, vals, allowed in (
            ("formats", self.formats, FORMATS),
            ("renders", self.renders, RENDERS),
            ("langs", self.langs, LANGS),
            ("controls", self.controls, CONTROLS),
        ):
            bad = [v for v in vals if v not in allowed]
            if bad:
                raise ValueError(f"unknown {name}: {bad} (allowed: {allowed})")

    @cached_property
    def build_combinations(self) -> list[list[str]]:
        return list(itertools.product(
            self.langs,
            self.renders,
            self.formats,
            self.controls,
        ))


class PromptBuilder:
    def __init__(self, config: BuildConfig | None = None) -> None:
        self.config = config or BuildConfig()

    def build(self, item: TestItem) -> list[PromptInstance]:
        cfg = self.config
        out: list[PromptInstance] = []

        for lang, render, fmt, control in cfg.build_combinations:
            if not item.has_lang(lang):
                if cfg.skip_unavailable:
                    continue
                raise BuildError(f"{item.item_id}: no {lang} block")

            if not self._render_available(item, lang, render):
                if cfg.skip_unavailable:
                    continue
                raise BuildError(
                    f"{item.item_id}: render {render!r} unavailable "
                    f"for lang {lang!r}"
                )

            if fmt == "meaning" and not item.meaning(lang):
                if cfg.skip_unavailable:
                    continue
                raise BuildError(
                    f"{item.item_id}: no meaning.{lang} "
                    "for the meaning format"
                )

            for order in self._orders(fmt):
                out.append(
                    self._make(item, lang, render, fmt, control, order)
                )

        return out

    def build_many(self, items: Iterable[TestItem]) -> list[PromptInstance]:
        out: list[PromptInstance] = []
        for it in items:
            out.extend(self.build(it))
        return out

    # internals
    @staticmethod
    def _render_available(item: TestItem, lang: str, render: str) -> bool:
        # if lang == "bn":
        #     return render == "bengali"
        if render == "respelled":
            return bool(item.proverb.respelled)
        return True

    def _orders(self, fmt: str) -> list[tuple[str, ...]]:
        keys = FORMAT_OPTIONS[fmt]
        if not self.config.all_orders:
            return [keys]
        if fmt == "none" and self.config.none_last:
            content = [k for k in keys if k != NONE_KEY]
            return [(*p, NONE_KEY) for p in itertools.permutations(content)]
        return list(itertools.permutations(keys))

    @staticmethod
    def proverb_surface(item: TestItem, lang: str, render: str) -> str:
        """How the proverb itself is shown for a (lang, render)."""
        p = item.proverb
        # if lang == "bn" or render == "bengali":
        #     return p.bengali
        if render == "bengali":
            return p.bengali
        if render == "romanised":
            return p.romanised
        if render == "gloss":
            core = p.romanised.strip().rstrip(_END_PUNCT).rstrip()
            return f'{core} ("{p.literal_gloss.strip().rstrip(_END_PUNCT)}")'
        if render == "gloss_only":
            return p.literal_gloss
        if render == "respelled":
            if not p.respelled:
                raise BuildError(f"{item.item_id}: no respelled form")
            return p.respelled
        raise BuildError(f"unknown render {render!r}")

    def render_context(self, item: TestItem, lang: str, render: str) -> str:
        block = item.block(lang)
        assert block is not None
        lines = []
        replaced = False
        for t in block.turns:
            text = t.text
            if lang == "en" and render != "romanised" and t.speaker == block.proverb_speaker and not replaced:
                span = find_span(text, item.proverb.romanised)
                if span is not None:
                    s, e = span
                    matched = text[s:e]
                    # keep the original sentence-final punctuation, whatever the surface form
                    trail = matched[len(matched.rstrip(_END_PUNCT)) :]
                    surface = self.proverb_surface(item, lang, render).strip().rstrip(_END_PUNCT).rstrip()
                    text = text[:s] + surface + trail + text[e:]
                    replaced = True
            if lang == "bn" and render != "bengali" and t.speaker == block.proverb_speaker and not replaced:
                span = find_span(text, item.proverb.bengali)
                if span is not None:
                    s, e = span
                    matched = text[s:e]
                    # keep the original sentence-final punctuation, whatever the surface form
                    trail = matched[len(matched.rstrip(_END_PUNCT)) :]
                    surface = self.proverb_surface(item, lang, render).strip().rstrip(_END_PUNCT).rstrip()
                    text = text[:s] + surface + trail + text[e:]
                    replaced = True
            lines.append(f"{t.speaker}: {text}")
        if lang == "en" and render != "romanised" and not replaced:
            raise BuildError(f"{item.item_id}: could not locate the romanised proverb in the speaker's turn")
        return "\n".join(lines)

    def _option_text(self, item: TestItem, lang: str, fmt: str, key: str) -> str:
        if key == NONE_KEY:
            return FIXED_TEXT[lang]["none_option"]
        block = item.block(lang)
        assert block is not None
        if fmt == "meaning" and key == "correct":
            meaning = item.meaning(lang)
            assert meaning
            return meaning
        return block.options.get(key)

    def _make(
        self,
        item: TestItem,
        lang: str,
        render: str,
        fmt: str,
        control: str,
        order: tuple[str, ...],
    ) -> PromptInstance:
        ft = FIXED_TEXT[lang]
        block = item.block(lang)
        assert block is not None

        if fmt == "meaning":
            head = f"{ft['proverb_label']} {self.proverb_surface(item, lang, render)}"
            question = ft["meaning_q"]
        else:
            head = self.render_context(item, lang, render)
            question = ft[f"{'negative' if fmt == 'negative' else 'normal'}_q"].format(speaker=block.proverb_speaker)

        texts = [self._option_text(item, lang, fmt, k) for k in order]
        letters = LETTERS[: len(order)]
        option_lines = "\n".join(f"{L}. {t}" for L, t in zip(letters, texts, strict=True))

        parts = []
        if self.config.instruction:
            parts.append(self.config.instruction)
        parts.append(head)
        q_line = "" if control == "no_question" else f"{ft['question_label']} {question}"
        body = f"{q_line}\n{option_lines}" if q_line else option_lines
        prompt = "\n\n".join(parts) + "\n\n" + body + f"\n{ft['answer_cue']}"

        cloze_parts = list(parts)
        cloze = "\n\n".join(cloze_parts)
        cloze += (f"\n\n{q_line}" if q_line else "") + f"\n{ft['answer_cue']}"

        gold = FORMAT_GOLD[fmt]
        iid = f"{item.item_id}|{lang}|{render}|{fmt}|{control}|{'>'.join(order)}"
        return PromptInstance(
            instance_id=iid,
            item_id=item.item_id,
            proverb_id=item.proverb.proverb_id,
            lang=lang,
            render=render,
            format=fmt,
            control=control,
            option_keys=list(order),
            option_texts=texts,
            gold_key=gold,
            prompt=prompt,
            cloze_prompt=cloze,
            meta={
                "figurative": item.proverb.figurative,
                "region": item.proverb.region,
                "has_false_friends": item.proverb.has_false_friends,
                "subsets": list(item.meta.subsets),
                "split": item.meta.split,
            },
        )


# --------------------------------------------------------------------------- training targets


def sft_target(inst: PromptInstance, style: str = "letter_text") -> str:
    """Completion string for SFT on a built instance.

    style: ``letter`` -> " A"; ``letter_text`` -> " A. <option text>"; ``text`` -> " <option text>".
    The leading space matches plain-text scoring after ``Answer:``.
    """
    L = inst.gold_letter
    t = inst.text_of(inst.gold_key)
    if style == "letter":
        return f" {L}"
    if style == "letter_text":
        return f" {L}. {t}"
    if style == "text":
        return f" {t}"
    raise ValueError(style)


def instances_by_item(instances: Iterable[PromptInstance]) -> dict[str, list[PromptInstance]]:
    out: dict[str, list[PromptInstance]] = {}
    for inst in instances:
        out.setdefault(inst.item_id, []).append(inst)
    return out
