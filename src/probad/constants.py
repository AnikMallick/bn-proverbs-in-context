"""Fixed vocabulary of the project: option keys, formats, renders, fixed question text, system prompts."""

OPTION_KEYS = ("correct", "literal", "plausible")
NONE_KEY = "none"  # the "None of these" option used by the `none` format

# normal   : What does X mean?        options {correct, literal}            gold correct
# negative : What does X NOT mean?    options {correct, literal}            gold literal
# none     : What does X mean?        options {literal, plausible, none}    gold none
# meaning  : What does this proverb mean? (no conversation) {meaning, literal}  gold correct
FORMATS = ("normal", "negative", "none", "meaning")
FORMAT_OPTIONS = {
    "normal": ("correct", "literal"),
    "negative": ("correct", "literal"),
    "none": ("literal", "plausible", NONE_KEY),
    "meaning": ("correct", "literal"),
}
FORMAT_GOLD = {"normal": "correct", "negative": "literal", "none": NONE_KEY, "meaning": "correct"}
FORMAT_CHANCE = {"normal": 0.5, "negative": 0.5, "none": 1 / 3, "meaning": 0.5}

# romanised : romanised proverb (default)
# gloss     : romanised proverb + ("literal English gloss")
# gloss_only: the literal English gloss replaces the proverb
# respelled : romanised proverb with false-friend words respelled
# bengali   : Bengali-script proverb
RENDERS = ("romanised", "gloss", "gloss_only", "respelled", "bengali")
LANGS = ("en", "bn")
CONTROLS = ("none", "no_question")  # no_question: question line removed (answer-option artefacts)
LETTERS = ("A", "B", "C", "D", "E")

FIXED_TEXT = {
    "en": {
        "normal_q": "What does {speaker} mean?",
        "negative_q": "What does {speaker} NOT mean?",
        "meaning_q": "What does this proverb mean?",
        "none_option": "None of these",
        "answer_cue": "Answer:",
        "question_label": "Question:",
        "proverb_label": "Proverb:",
    },
    "bn": {
        "normal_q": "{speaker} কী বোঝাতে চাইছে?",
        "negative_q": "{speaker} কী বোঝাতে চাইছে না?",
        "meaning_q": "এই প্রবাদটির অর্থ কী?",
        "none_option": "এগুলোর কোনোটিই নয়",
        "answer_cue": "উত্তর:",
        "question_label": "প্রশ্ন:",
        "proverb_label": "প্রবাদ:",
    },
}

# Prompt-level conditions. C0 = no system prompt. Freeze these before the final runs.
SYSTEM_PROMPTS = {
    "C0": None,
    "C1a": "You are a native Bengali speaker who grew up in Bengal.",
    "C1b": (
        "You are a native Bengali speaker who grew up in a middle-class family in Kolkata and has "
        "family in rural Bengal and Bangladesh. You speak Bengali at home and English at college and "
        "work, and you often switch between the two. You know Bengali proverbs (probad) from your "
        "grandparents, from school and from everyday conversation, and you know that people use them "
        "to comment on a situation, tease, warn or give advice rather than to talk about the literal "
        "objects in them. The people in the conversations you read are Bengali friends, family "
        "members, classmates and colleagues talking about ordinary life."
    ),
}

# STOPWORD Path
BN_STOPWORD_PATH = r"stopwords-bn/stopwords-bn.json"