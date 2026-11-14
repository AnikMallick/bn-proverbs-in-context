import json
import random
import re

PUNCT = ".,!?;:।'\"‘’“”"

# MAPS exp
QUESTIONS = {
    "pos":  "What does the person mean by the proverb?",
    "neg":  "What does the person not mean by the proverb?",
    "neg1": "Which answer is contrary to what the person means by the proverb?",
    "neg2": "Which answer is impossible as the interpretation of what the person means by the proverb?",
    "neg3": "Pick the opposite answer to what the person means by the proverb.",
    "neg4": "Pick the wrong answer to what the person means by the proverb.",
}

MEM_TEMPLATES = {
    "en": ["proverb: {p}", "complete this proverb: {p}", "finish the proverb: {p}",
           "what's missing at the end of this proverb: {p}", "what's the last word of this proverb: {p}"],
    "bn": ["প্রবাদ: {p}", "এই প্রবাদটি সম্পূর্ণ করো: {p}", "প্রবাদটি শেষ করো: {p}",
           "এই প্রবাদের শেষে কি অনুপস্থিত: {p}", "এই প্রবাদের শেষ শব্দ কি: {p}"],
}

def clean(text):
    return text.strip().strip(PUNCT).strip()


def context(item, lang="bn"):
    return " ".join(f"{t['speaker']}: {t['text']}" for t in item[lang]["turns"])


def proverb(item, lang="bn"):
    return clean(item["proverb"]["bengali"] if lang == "bn" else item["proverb"]["romanised"])


def options(item, lang, keys, seed=0):
    """Shuffle option keys per item (reproducible); return (choices string, letter of each key)."""
    keys = list(keys)
    random.Random(f"{item['item_id']}|{seed}").shuffle(keys)
    letters = "abc"[: len(keys)]
    choices = " ".join(f"{L}: {item[lang]['options'][k]}" for L, k in zip(letters, keys))
    return choices, dict(zip(keys, letters))


def mcq(item, lang="bn", question="pos", keys=("correct", "plausible"), seed=0, proverb_text=None, ctx=None):
    """MAPS prompt. Returns (prompt, answer letter)."""
    choices, letter = options(item, lang, keys, seed)
    p = proverb_text or proverb(item, lang)
    c = ctx or context(item, lang)
    q = f"Question: {QUESTIONS[question]}\n" if question else ""
    prompt = f"{q}Proverb:'{p}'\nContext: {c}\nChoices: {choices}\nAnswer:".lower()
    gold = "correct" if question in ("pos", None) else [k for k in keys if k != "correct"][0]
    return prompt, letter[gold], list("abcd"[:len(letter.keys())])


# maps experiments
def e1_memorisation(item, render="bengali", template=0):
    """Knowing: proverb minus last word. Returns (prompt, missing word), or None for one-word proverbs."""
    full = clean(item["proverb"]["bengali"] if render == "bengali" else item["proverb"]["romanised"])
    if " " not in full:
        return None
    front, last = full.rsplit(" ", 1)
    lang = "bn" if render == "bengali" else "en"
    return MEM_TEMPLATES[lang][template].format(p=front).lower(), last.lower()


def e2_reasoning(item, seed=0, keys=("correct", "plausible")):
    """Using: main MAPS task. keys=("correct","literal") or 3 keys for your extensions."""
    return mcq(item, "bn", "pos", keys, seed)


def e5_negated(item, question="neg", seed=0):
    """Same task, asks for the wrong answer. question: neg, neg1..neg4."""
    return mcq(item, "bn", question, ("correct", "plausible"), seed)


def e5_control(item, seed=0):
    """No question line; answer is still 'correct' (expect ~chance)."""
    return mcq(item, "bn", None, ("correct", "plausible"), seed)


def e6_translation(item, condition="en_gloss", seed=0):
    """bn_native | en_romanised | en_gloss (proverb replaced by its literal English gloss)."""
    if condition == "bn_native":
        return mcq(item, "bn", seed=seed)
    if condition == "en_romanised":
        return mcq(item, "en", seed=seed)
    gloss = clean(item["proverb"]["literal_gloss"])
    rom = clean(item["proverb"]["romanised"])
    ctx = re.sub(re.escape(rom), gloss, context(item, "en"), flags=re.I)
    return mcq(item, "en", seed=seed, proverb_text=gloss, ctx=ctx)


def e7_few_shot(item, demo_items, k=2, seed=0):
    """k demos (from val, never test) + the test prompt."""
    rng = random.Random(seed)
    demos = [e2_reasoning(d, seed=rng.random()) for d in rng.sample(demo_items, k)]
    prompt, gold, all_options = e2_reasoning(item, seed=seed)
    return "\n".join(f"{p} {a}" for p, a, o in demos) + "\n" + prompt, gold, all_options


def e9_template(item, template=1, seed=0):
    """Appendix templates 1-3 (check wording against the paper)."""
    choices, letter = options(item, "bn", ("correct", "plausible"), seed)
    p, c = proverb(item), context(item)
    t = {
        1: f"Proverb:'{p}'\nContext: {c}\nChoices: {choices}\nQuestion: {QUESTIONS['pos']}\nAnswer:",
        2: f"Question: What is a probable interpretation of this proverb?\nProverb:'{p}'\nContext: {c}\n"
           f"Choices: {choices}\nPlease choose between A and B.\nAnswer:",
        3: f"Question: How would one interpret this proverb given the context?\nProverb:'{p}'\nContext: {c}\n"
           f"Choices: {choices}\nAnswer:",
    }[template]
    return t.lower(), letter["correct"], list("abcd"[:len(letter.keys())])

# my experiment
OWN = {
    "en": {"q": "Question:", "mean": "What does {s} mean?", "not_mean": "What does {s} NOT mean?",
           "proverb_q": "What does this proverb mean?", "proverb": "Proverb:", "none": "None of these",
           "answer": "Answer:"},
    "bn": {"q": "প্রশ্ন:", "mean": "{s} কী বোঝাতে চাইছে?", "not_mean": "{s} কী বোঝাতে চাইছে না?",
           "proverb_q": "এই প্রবাদটির অর্থ কী?", "proverb": "প্রবাদ:", "none": "এগুলোর কোনোটিই নয়",
           "answer": "উত্তর:"},
}

SYSTEM_PROMPTS = {
    "C0": None,
    "C1a": "You are a native Bengali speaker who grew up in Bengal.",
    "C1b": ("You are a native Bengali speaker who grew up in a middle-class family in Kolkata and has "
            "family in rural Bengal and Bangladesh. You speak Bengali at home and English at college and "
            "work, and you often switch between the two. You know Bengali proverbs (probad) from your "
            "grandparents, from school and from everyday conversation, and you know that people use them "
            "to comment on a situation, tease, warn or give advice rather than to talk about the literal "
            "objects in them. The people in the conversations you read are Bengali friends, family "
            "members, classmates and colleagues talking about ordinary life."),
}


def surface(item, render):
    """How the proverb is shown: romanised | gloss | gloss_only | respelled | bengali."""
    p = item["proverb"]
    rom, gloss = clean(p["romanised"]), clean(p["literal_gloss"])
    return {"romanised": rom, "gloss": f'{rom} ("{gloss}")', "gloss_only": gloss,
            "respelled": clean(p["respelled"]) if p.get("respelled") else None,
            "bengali": clean(p["bengali"])}[render]


def conversation(item, lang="en", render=None):
    """Conversation lines; for English, optionally swap the romanised proverb for another render."""
    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in item[lang]["turns"])
    if lang == "en" and render and render != "romanised":
        text = re.sub(re.escape(clean(item["proverb"]["romanised"])), surface(item, render), text, flags=re.I)
    return text


def own_mcq(item, lang, opts, gold, head, question, seed=0, none_last=False):
    """opts: {key: text}. Returns (prompt, gold letter)."""
    keys = list(opts)
    if none_last:
        rest = [k for k in keys if k != "none"]
        random.Random(f"{item['item_id']}|{seed}").shuffle(rest)
        keys = rest + ["none"]
    else:
        random.Random(f"{item['item_id']}|{seed}").shuffle(keys)
    t = OWN[lang]
    lines = "\n".join(f"{L}. {opts[k]}" for L, k in zip("ABC", keys))
    q = f"{t['q']} {question}\n" if question else ""
    prompt = f"{head}\n\n{q}{lines}\n{t['answer']}"
    return prompt, "abc"[keys.index(gold)], list("abcd"[:len(keys)])


def x1_correct_vs_literal(item, lang="en", render="romanised", seed=0):
    """Main own task: does the model pick the meaning over the word-for-word reading?"""
    o = item[lang]["options"]
    s = item[lang]["proverb_speaker"]
    return own_mcq(item, lang, {"correct": o["correct"], "literal": o["literal"]}, "correct",
                   conversation(item, lang, render), OWN[lang]["mean"].format(s=s), seed)


def x2_negative(item, lang="en", seed=0):
    """'What does X NOT mean?' over correct vs literal -> answer is the literal option."""
    o = item[lang]["options"]
    s = item[lang]["proverb_speaker"]
    return own_mcq(item, lang, {"correct": o["correct"], "literal": o["literal"]}, "literal",
                   conversation(item, lang), OWN[lang]["not_mean"].format(s=s), seed)


def x3_none_of_these(item, lang="en", seed=0):
    """Correct option removed; literal, plausible, 'None of these' -> answer is 'None of these'."""
    o = item[lang]["options"]
    s = item[lang]["proverb_speaker"]
    opts = {"literal": o["literal"], "plausible": o["plausible"], "none": OWN[lang]["none"]}
    return own_mcq(item, lang, opts, "none", conversation(item, lang), OWN[lang]["mean"].format(s=s),
                   seed, none_last=True)


def x4_meaning_no_context(item, lang="en", seed=0):
    """Knowing without context: proverb alone, stated meaning vs literal option."""
    meaning = item["proverb"]["meaning"][lang]
    head = f"{OWN[lang]['proverb']} {item['proverb']['romanised'] if lang == 'en' else item['proverb']['bengali']}"
    opts = {"correct": meaning, "literal": item[lang]["options"]["literal"]}
    return own_mcq(item, lang, opts, "correct", head, OWN[lang]["proverb_q"], seed)


def x5_three_way(item, lang="en", seed=0):
    """Correct vs literal vs plausible (chance = 1/3)."""
    o = item[lang]["options"]
    s = item[lang]["proverb_speaker"]
    return own_mcq(item, lang, dict(o), "correct", conversation(item, lang), OWN[lang]["mean"].format(s=s), seed)


def x6_render(item, render="gloss", seed=0):
    """English conversation with the proverb shown as romanised | gloss | gloss_only | respelled | bengali.
    Returns None if the item has no respelled form."""
    if surface(item, render) is None:
        return None
    return x1_correct_vs_literal(item, "en", render, seed)


def x7_no_question(item, lang="en", seed=0):
    """Control: question line removed (answer-option artefacts)."""
    o = item[lang]["options"]
    return own_mcq(item, lang, {"correct": o["correct"], "literal": o["literal"]}, "correct",
                   conversation(item, lang), None, seed)


def x8_system_prompt(item, condition="C1b", lang="en", seed=0, system=None):
    """C0 / C1a / C1b persona prompts; pass system=<GEPA best prompt> for C1c.
    Returns (system, prompt, gold)."""
    prompt, gold, all_options = x1_correct_vs_literal(item, lang, seed=seed)
    return (system if system is not None else SYSTEM_PROMPTS[condition]), prompt, gold, all_options


def x9_false_friend_pair(item, seed=0):
    """Same item twice: romanised vs respelled (only items with false friends and a respelling)."""
    if not item["proverb"].get("false_friends") or not item["proverb"].get("respelled"):
        return None
    return {"romanised": x6_render(item, "romanised", seed), "respelled": x6_render(item, "respelled", seed)}


def quadrant(memorised, correct):
    """Knowing vs using per proverb: know+use, know-only, use-only, neither."""
    return {(True, True): "know+use", (True, False): "know_only",
            (False, True): "use_only", (False, False): "neither"}[(bool(memorised), bool(correct))]