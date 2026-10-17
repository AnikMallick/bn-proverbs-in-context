"""Letter log-prob scoring with a Hugging Face causal LM, one prompt at a time.

For each prompt instance: log P(" A"), log P(" B"), ... as the next token after "Answer:".
The probabilities are renormalised over the letters shown and stored per *option key*
(p_correct, p_literal, ...), so averaging over option orders later is a plain groupby.
"""

import math
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probad.constants import LETTERS

INSTANCE_COLS = {"instance_id", "item_id", "proverb_id", "lang", "render", "format", "control", "gold_key"}


def load_model(name, dtype=None):
    """Load a model on CUDA (Kaggle), MPS (Mac) or CPU.

    Default dtype: bfloat16 on Ampere+ GPUs, else float32 (a T4 has no bf16, and fp16 can overflow).
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok


def format_prompt(tok, prompt, system=None, chat=False):
    """Base models: plain text with the system prompt on top. Instruct models (chat=True): chat template."""
    if not chat:
        return f"{system}\n\n{prompt}" if system else prompt
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def next_token_logprobs(model, tok, text, chat=False):
    """Log-probabilities over the vocabulary for the token after `text`."""
    enc = tok(text, return_tensors="pt", add_special_tokens=not chat).to(model.device)  # chat templates include BOS
    logp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), dim=-1)
    if torch.isnan(logp).any():
        raise ValueError("NaN log-probs: the model overflowed; load it in float32")
    return logp


def letter_ids(tok, chat=False):
    """Token id of each answer letter: " A" after "Answer:" (plain), "A" right after a chat template."""
    ids = {}
    for letter in LETTERS:
        enc = tok.encode(letter if chat else " " + letter, add_special_tokens=False)
        assert len(enc) == 1, f"{letter!r} is not a single token for this tokenizer"
        ids[letter] = enc[0]
    return ids


def score(model, tok, instances, condition="C0", system=None, chat=False, model_name=None, out=None):
    """One row per instance. If `out` already exists it is loaded instead of re-scoring."""
    if out and Path(out).exists():
        return pd.read_json(out, lines=True)
    ids = letter_ids(tok, chat)
    rows = []
    for inst in tqdm(instances, desc=condition):
        logp = next_token_logprobs(model, tok, format_prompt(tok, inst.prompt, system, chat), chat)
        lp = {L: logp[ids[L]].item() for L in inst.letters}
        mass = sum(math.exp(v) for v in lp.values())  # total probability on the shown letters
        gold, pred = inst.gold_letter, max(lp, key=lp.get)
        rows.append(
            {
                "model": model_name or model.name_or_path,
                "condition": condition,
                **inst.model_dump(include=INSTANCE_COLS),
                **inst.meta,
                "order": ">".join(inst.option_keys),
                **{f"lp_{k}": lp[L] for L, k in zip(inst.letters, inst.option_keys)},
                **{f"p_{k}": math.exp(lp[L]) / mass for L, k in zip(inst.letters, inst.option_keys)},
                "pred_key": inst.key_of(pred),
                "correct": pred == gold,
                "margin": lp[gold] - max(v for L, v in lp.items() if L != gold),
                "letter_mass": mass,
            }
        )
    df = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out, orient="records", lines=True, force_ascii=False)
    return df
