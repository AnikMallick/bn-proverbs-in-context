PROVERBS
────────────────────
proverb_id PK
bengali
romanised
literal_gloss
meaning
figurative
region
in_maps
cultural_note


CONTEXTS
────────────────────
context_id PK
proverb_id FK ──────────> PROVERBS.proverb_id
turns
proverb_speaker


ITEMS
────────────────────
item_id PK
proverb_id FK ──────────> PROVERBS.proverb_id
context_id FK ──────────> CONTEXTS.context_id
options
memorisation
meta


{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Proverb Dataset Item",
  "type": "object",
  "required": [
    "item_id",
    "proverb_id",
    "context_id",
    "options",
    "memorisation",
    "meta"
  ],
  "properties": {
    "item_id": {
      "type": "string",
      "description": "Unique identifier for the dataset item"
    },
    "proverb_id": {
      "type": "string",
      "description": "Foreign key referencing the proverb"
    },
    "context_id": {
      "type": "string",
      "description": "Foreign key referencing the context"
    },
    "options": {
      "type": "object",
      "required": [
        "correct",
        "literal",
        "plausible"
      ],
      "properties": {
        "correct": {
          "type": "string"
        },
        "literal": {
          "type": "string"
        },
        "plausible": {
          "type": "string"
        }
      },
      "additionalProperties": false
    },
    "memorisation": {
      "type": "object",
      "required": [
        "prefix",
        "target"
      ],
      "properties": {
        "prefix": {
          "type": "string"
        },
        "target": {
          "type": "string"
        }
      },
      "additionalProperties": false
    },
    "meta": {
      "type": "object",
      "required": [
        "author",
        "created",
        "split",
        "version"
      ],
      "properties": {
        "author": {
          "type": "string"
        },
        "created": {
          "type": "string",
          "format": "date"
        },
        "annotator2_label": {
          "type": [
            "string",
            "null"
          ]
        },
        "agreement": {
          "type": [
            "number",
            "null"
          ]
        },
        "stereotype_flag": {
          "type": "boolean"
        },
        "split": {
          "type": "string",
          "enum": [
            "train",
            "validation",
            "test"
          ]
        },
        "version": {
          "type": "integer",
          "minimum": 1
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}


'''
{
  "item_id": "bn_0001",
  "proverb_id": "prv_017",
  "proverb": {
    "bengali": "ঠাকুর ঘরে কে? আমি কলা খাই না",
    "romanised": "Thakur ghore ke? Ami kola khai na.",
    "literal_gloss": "Who is in the prayer room? I don't eat bananas.",
    "meaning": "A guilty person gives themselves away by denying something no one accused them of.",
    "figurative": true,
    "region": "both",
    "in_maps": false,
    "cultural_note": "Bananas are common offerings in a household shrine (thakur ghor); the 'thief' denies eating them before anyone asks."
  },
  "context": {
    "turns": [
      {"speaker": "Ma", "text": "Someone finished the rosogollas I was saving for the guests."},
      {"speaker": "Rahul", "text": "Don't look at me, I haven't even opened the fridge today!"},
      {"speaker": "Didi", "text": "Thakur ghore ke? Ami kola khai na."}
    ],
    "proverb_speaker": "Didi"
  },
  "options": {
    "correct": "Rahul's eager denial, before anyone blamed him, shows that he ate the sweets.",
    "literal": "Someone in the prayer room is saying they do not eat bananas.",
    "plausible": "The sweets were probably taken as an offering to the household deity."
  },
  "memorisation": {"prefix": "Thakur ghore ke? Ami", "target": "kola khai na"},
  "meta": {"author": "you", "created": "2026-10-12", "annotator2_label": null,
           "agreement": null, "stereotype_flag": false, "split": "test", "version": 1}
}
'''