"""Score the simple_maps.py experiments with a Hugging Face causal LM.

Multiple choice: log P(letter) as the next token after "Answer:"/"answer:"/"উত্তর:", over the letters shown.
Memorisation (e1): greedy generation.

    from simple_maps import *
    model, tok = load_model("allenai/OLMo-2-0425-1B")
    test, maps, val = load("test.jsonl"), load("maps_clean.jsonl"), load("val.jsonl")

    e2 = score(model, tok, maps, e2_reasoning, label="E2", out="results/olmo1b/e2.jsonl")
    e6 = score(model, tok, maps, e6_translation, label="E6_gloss", condition="en_gloss")
    e7 = score(model, tok, maps, e7_few_shot, label="E7_k5", demo_items=val, k=5)
    x1 = score(model, tok, test, x1_correct_vs_literal, label="x1_bn", lang="bn")
    x8 = score(model, tok, test, x8_system_prompt, label="x8_C1b", condition="C1b")   # system comes from x8
    x9 = score(model, tok, test, x9_false_friend_pair, label="x9")                    # one row per variant
    x1_chat = score(model, tok, test, x1_correct_vs_literal, chat=True)               # instruct models

    mem = pd.concat([generate_all(model, tok, maps, e1_memorisation, render=r, template=t)
                     for r in ("bengali", "romanised") for t in range(5)])
    memorised = mem.groupby(["proverb_id", "render"]).correct.any()                   # MAPS: any template

    e2.correct.mean(); e2.groupby("figurative").correct.mean()                        # E2, E4
"""

import math
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import queue
import threading

OPTIONS = ["a", "b", "c", "d", "e"]
PUNCT = ".,!?;:।'\"‘’“”"

# model - single
def load_model(name, dtype=None):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tok

# model with pool -> to be used with thread pool
# from concurrent.futures import ThreadPoolExecutor <- usage

# def process_item(item):
#     r = exp.e5_negated(item.model_dump(), question="neg")

#     return answer_one(
#         model, tok,
#         r[0], r[1], r[2],
#         system=None,
#         chat=False
#     )


# with ThreadPoolExecutor(max_workers=2) as executor:
#     results = list(executor.map(process_item, train_data[:10]))

# print(results)



class LoadBalancedModel:

    def __init__(self, name, gpu_ids=(0, 1), dtype=None):

        self.models = []
        self.devices = []
        self.pool = queue.Queue()
        self.local = threading.local()
        self.name = name

        if dtype is None:
            dtype = torch.float16

        self.tok = AutoTokenizer.from_pretrained(name)

        for gpu_id in gpu_ids:
            device = f"cuda:{gpu_id}"

            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=dtype
            ).to(device).eval()

            self.models.append(model)
            self.devices.append(device)
            self.pool.put(len(self.models) - 1)

    @property
    def device(self):
        # Reserve a free model for this thread
        idx = self.pool.get()
        self.local.idx = idx
        return self.devices[idx]

    def _release(self):
        idx = getattr(self.local, "idx", None)

        if idx is not None:
            del self.local.idx
            self.pool.put(idx)

    def __call__(self, *args, **kwargs):
        idx = getattr(self.local, "idx", None)

        if idx is None:
            idx = self.pool.get()

        try:
            return self.models[idx](*args, **kwargs)
        finally:
            self.local.idx = None
            self.pool.put(idx)

    def generate(self, *args, **kwargs):
        idx = getattr(self.local, "idx", None)

        if idx is None:
            idx = self.pool.get()

        try:
            return self.models[idx].generate(*args, **kwargs)
        finally:
            self.local.idx = None
            self.pool.put(idx)


def format_prompt(tok, prompt, system=None, chat=False):
    """Base models: plain text with the system prompt on top. Instruct models (chat=True): chat template."""
    if not chat:
        return f"{system}\n\n{prompt}" if system else prompt
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def next_token_logprobs(model, tok, text, chat=False):
    enc = tok(text, return_tensors="pt", add_special_tokens=not chat).to(model.device)  # chat templates add BOS
    logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
    if torch.isnan(logp).any():
        raise ValueError("NaN log-probs: the model overflowed; load it in float32")
    return logp


def letter_id(tok, letter, chat=False):
    """Token for the letter: ' a' after 'answer:' (plain text), 'a' right after a chat template.
    If the tokenizer splits it, the first piece is used."""
    ids = tok.encode(letter if chat else " " + letter, add_special_tokens=False)
    return ids[-1] if len(ids) > 1 and tok.decode(ids[:1]).strip() == "" else ids[0]


def item_info(it):
    return {"item_id": it["item_id"], "proverb_id": it["proverb"]["proverb_id"],
            "figurative": it["proverb"]["figurative"], "split": it["meta"]["split"],
            "subsets": it["meta"].get("subsets", [])}

# mcq
def answer_one(model, tok, prompt: str, gold:str, options: list[str], system=None, chat=False):
    """Score one MCQ prompt: log P of each shown letter as the next token."""
    logp = next_token_logprobs(model, tok, format_prompt(tok, prompt, system, chat), chat)
    lp = {L: logp[letter_id(tok, L, chat)].item() for L in options}
    mass = sum(math.exp(v) for v in lp.values())
    pred = max(lp, key=lp.get)
    return {"gold": gold.lower(), "pred": pred, "correct": pred.lower() == gold.lower(),
            **{f"p_{L.lower()}": math.exp(v) / mass for L, v in lp.items()},
            "margin": lp[gold] - max(v for L, v in lp.items() if L != gold.lower()),
            "letter_mass": mass}

# memorization
@torch.inference_mode()
def generate(model, tok, prompt:str, max_new_tokens=4):
    """Greedy continuation (MAPS used max_new_tokens=4)."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


# validators
def check_choice(output, gold):
    """The model's first token is the gold letter (" a", "B.", "(c)" all count)."""
    m = re.match(r"\W*([abc])\b", output.strip().lower())
    return bool(m) and m.group(1) == gold


def check_memorisation(output, missing_word):
    """Output starts with (or contains) the missing last word."""
    out = " ".join(output.lower().translate(str.maketrans("", "", PUNCT)).split())
    return missing_word.strip(PUNCT) in out.split()[:3] if out else False


def check_memorisation_graded(output, missing_word):
    """0-1 partial credit: best character similarity between the missing word and any of the first 3 words."""
    from difflib import SequenceMatcher
    words = output.lower().translate(str.maketrans("", "", PUNCT)).split()[:3]
    target = missing_word.strip(PUNCT).lower()
    return max((SequenceMatcher(None, target, w).ratio() for w in words), default=0.0)

# from copy import copy
# def run(items, make, llm, check=check_choice, system=None, **kw):
#     """make(item, **kw) -> (prompt, gold); llm(prompt, system) -> text.
#     Returns one record per item; accuracy = mean of record["correct"]."""
#     records = []
#     for it in items:
#         built = make(it, **kw)
#         if built is None:                      # e.g. no respelling / one-word proverb
#             continue
#         prompt, gold = built
#         output = llm(prompt, system)
#         records.append({"item_id": it["item_id"], "proverb_id": it["proverb"]["proverb_id"],
#                         "figurative": it["proverb"]["figurative"], "prompt": prompt, "gold": gold,
#                         "output": output, "correct": float(check(output, gold)), "correct_choice": check.__name__})
#     return records

# def quadrant(memorised, correct):
#     """Knowing vs using per proverb: know+use, know-only, use-only, neither."""
#     return {(True, True): "know+use", (True, False): "know_only",
#             (False, True): "use_only", (False, False): "neither"}[(bool(memorised), bool(correct))]

# def accuracy(records, key=None):
#     """Overall accuracy, or per group if key is given (e.g. key="figurative")."""
#     if key is None:
#         return sum(r["correct"] for r in records) / max(len(records), 1)
#     groups = {}
#     for r in records:
#         groups.setdefault(r[key], []).append(r["correct"])
#     return {g: sum(v) / len(v) for g, v in groups.items()}

# def run_only(items, make, llm, system=None, **kw):
#     """make(item, **kw) -> (prompt, gold); llm(prompt, system) -> text.
#     Returns one record per item; accuracy = mean of record["correct"]."""
#     records = []
#     for it in items:
#         built = make(it, **kw)
#         if built is None:                      # e.g. no respelling / one-word proverb
#             continue
#         prompt, gold = built
#         output = llm(prompt, system)
#         records.append({"item_id": it["item_id"], "proverb_id": it["proverb"]["proverb_id"],
#                         "figurative": it["proverb"]["figurative"], "prompt": prompt, "gold": gold,
#                         "output": output})
#     return records

# def check(record, check: Callable[[str,str], float | bool]=check_choice):
#     r = copy(record)
#     r["correct"] = float(check(output, gold))
#     r["correct_choice"] = check.__name__
#     return r