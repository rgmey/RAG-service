# eval/dataset.py
"""
A small synthetic corpus + question set for retrieval evaluation.

Deliberately mixes two question styles:
  - "exact-term" questions (product codes, error codes, policy IDs) that
    vector search tends to miss, since a made-up name has little semantic
    content to embed near.
  - "paraphrase" questions that never use the source document's actual
    keywords, which keyword search tends to miss for the same reason in
    reverse.

This split is what lets the eval actually demonstrate hybrid search's
value, rather than just re-confirming that vector search alone works.
"""

CORPUS = [
    {
        "id": "eval_photosynthesis",
        "text": (
            "Photosynthesis is the process plants use to convert light energy, "
            "water, and carbon dioxide into glucose and oxygen. It occurs mainly "
            "in the chloroplasts of leaf cells, using the green pigment "
            "chlorophyll to capture sunlight."
        ),
        "style": "paraphrase",
    },
    {
        "id": "eval_zenith9",
        "text": (
            "The Zenith-9 turbine is a compact wind turbine rated for 9 kilowatts "
            "of peak output, designed for small-scale residential installations "
            "in low-wind coastal regions."
        ),
        "style": "exact-term",
    },
    {
        "id": "eval_hr114",
        "text": (
            "Policy HR-114 covers the company's remote work stipend, providing "
            "up to $500 per year toward home office equipment for employees "
            "working remotely more than three days per week."
        ),
        "style": "exact-term",
    },
    {
        "id": "eval_reef",
        "text": (
            "The Great Barrier Reef, stretching over 2,300 kilometers off the "
            "coast of Queensland, is the largest living structure on Earth and "
            "is visible from space. It is composed of billions of tiny coral "
            "organisms."
        ),
        "style": "paraphrase",
    },
    {
        "id": "eval_e4021",
        "text": (
            "Error E-4021 indicates that the device's firmware failed a checksum "
            "validation during boot. Reflashing the firmware from a verified "
            "source usually resolves it."
        ),
        "style": "exact-term",
    },
    {
        "id": "eval_suez",
        "text": (
            "The Suez Canal is an artificial waterway connecting the "
            "Mediterranean Sea to the Red Sea, allowing ships to avoid sailing "
            "around the southern tip of Africa. It opened in 1869."
        ),
        "style": "paraphrase",
    },
    {
        "id": "eval_rate429",
        "text": (
            "When a client exceeds the API's request quota, the server responds "
            "with HTTP 429 and a Retry-After header indicating how long to wait "
            "before sending another request."
        ),
        "style": "exact-term",
    },
    {
        "id": "eval_compound_interest",
        "text": (
            "Compound interest is calculated using A = P(1 + r/n)^(nt), where P "
            "is the principal, r the annual rate, n the number of compounding "
            "periods per year, and t the number of years."
        ),
        "style": "paraphrase",
    },
]

QUESTIONS = [
    {"question": "What does the Zenith-9 turbine do?", "gold_id": "eval_zenith9"},
    {
        "question": "How do plants turn sunlight into food?",
        "gold_id": "eval_photosynthesis",
    },
    {"question": "What does policy HR-114 cover?", "gold_id": "eval_hr114"},
    {
        "question": "What's the largest living structure visible from space?",
        "gold_id": "eval_reef",
    },
    {"question": "What causes error E-4021?", "gold_id": "eval_e4021"},
    {
        "question": "Which waterway lets ships skip sailing around Africa?",
        "gold_id": "eval_suez",
    },
    {
        "question": "What happens when I go over my API request quota?",
        "gold_id": "eval_rate429",
    },
    {
        "question": "What's the formula for compound interest?",
        "gold_id": "eval_compound_interest",
    },
]
