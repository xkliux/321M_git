import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).parent
DEVICE = "cpu"

#log config files
def load_json(name):
    with open(MODEL_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_json("config.json")

ENCODER_NAME = CONFIG["encoder_name"]
D_TEXT = int(CONFIG["embedding_dim"])
D_EXTRA = int(CONFIG["d_extra"])
EXTRA_FEATURES = CONFIG["extra_features"]

GLOBAL_MEAN = float(load_json("global_mean.json")["global_mean"])
SUBJECT_MEAN = load_json("subject_mean.json")
BENCHMARK_MEAN = load_json("benchmark_mean.json")
CONDITION_MEAN = load_json("condition_mean.json")
ITEM_MEAN = load_json("item_mean.json")
SO_PRIORS = load_json("so_priors.json")
ENCODER = SentenceTransformer(ENCODER_NAME)

ENCODE_CACHE = {}

#for speed
def cached_encode(text):
    text = str(text)
    if text not in ENCODE_CACHE:
        ENCODE_CACHE[text] = ENCODER.encode(
            text,
            normalize_embeddings=True,
        )
    return ENCODE_CACHE[text]

#model
class NCF(nn.Module):
    def __init__(self, d_text, d_extra):
        super().__init__()
        input_dim = 3 * d_text + d_extra
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, text_x, extra):
        x = torch.cat([text_x, extra], dim=1)
        return self.net(x).squeeze(-1)

MODEL = NCF(
    d_text=D_TEXT,
    d_extra=D_EXTRA,
).to(DEVICE)

MODEL.load_state_dict(
    torch.load(MODEL_DIR / "ncf_head.pt", map_location=DEVICE)
)
#set eval mode
MODEL.eval()

#helpers
def clip_prob(p):
    return float(max(0.05, min(0.95, p)))

def get_input_value(input_dict, key, default=""):
    val = input_dict.get(key, default)
    if val is None:
        return default
    return str(val)

#run encoder
def encode_texts(subject_text, item_text, benchmark_text):
    u = cached_encode(subject_text)
    v = cached_encode(item_text)
    b = cached_encode(benchmark_text)

    U = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
    V = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    B = torch.tensor(b, dtype=torch.float32).unsqueeze(0)

    X_text = torch.cat([U, V, B], dim=1)

    return U, V, B, X_text

#extract features
def build_extra_for_one(input_dict, U, V, B):

    #extract features
    subject_text = get_input_value(input_dict, "subject_content", "")
    item_text = get_input_value(input_dict, "item_content", "")
    # subject_text = get_input_value(input_dict, "subject_id", "")
    # item_text = get_input_value(input_dict, "item_id", "")

    benchmark_text = get_input_value(input_dict, "benchmark", "")
    condition = get_input_value(input_dict, "condition", "none")

    raw = {}
    #grab data from json, if not in json, fallback to global average
    raw["subject_prior"] = float(SUBJECT_MEAN.get(subject_text, GLOBAL_MEAN))
    raw["benchmark_prior"] = float(BENCHMARK_MEAN.get(benchmark_text, GLOBAL_MEAN))
    raw["condition_prior"] = float(CONDITION_MEAN.get(condition, GLOBAL_MEAN))
    raw["item_prior"] = float(ITEM_MEAN.get(item_text, GLOBAL_MEAN))

    # raw["benchmark_condition_prior"] = float(
    #     SO_PRIORS["benchmark_condition_prior"].get(
    #         f"{benchmark_text}|||{condition}",
    #         GLOBAL_MEAN,
    #     )
    # )
    #
    # raw["subject_benchmark_prior"] = float(
    #     SO_PRIORS["subject_benchmark_prior"].get(
    #         f"{subject_text}|||{benchmark_text}",
    #         GLOBAL_MEAN,
    #     )
    # )
    #
    # raw["subject_condition_prior"] = float(
    #     SO_PRIORS["subject_condition_prior"].get(
    #         f"{subject_text}|||{condition}",
    #         GLOBAL_MEAN,
    #     )
    # )

    #gaps
    raw["subject_minus_item"] = raw["subject_prior"] - raw["item_prior"]
    raw["subject_minus_benchmark"] = raw["subject_prior"] - raw["benchmark_prior"]
    #length
    item_len = min(len(item_text), 2000) / 2000.0
    subject_len = min(len(subject_text), 1000) / 1000.0
    benchmark_len = min(len(benchmark_text), 1000) / 1000.0

    raw["item_len"] = item_len
    raw["subject_len"] = subject_len
    raw["benchmark_len"] = benchmark_len
    raw["item_subject_len_ratio"] = item_len / (subject_len + 1e-6)
    #simarility
    raw["cos_subject_item"] = F.cosine_similarity(U, V, dim=1).item()
    raw["cos_subject_benchmark"] = F.cosine_similarity(U, B, dim=1).item()
    raw["cos_item_benchmark"] = F.cosine_similarity(V, B, dim=1).item()

    #build feature tensor
    values = [raw.get(name, 0.0) for name in EXTRA_FEATURES]
    extra = torch.tensor([values], dtype=torch.float32)
    return extra

#run predict with input
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    subject_text = get_input_value(input, "subject_content", "")
    item_text = get_input_value(input, "item_content", "")
    benchmark_text = get_input_value(input, "benchmark", "")
    condition = get_input_value(input, "condition", "none")

    U, V, B, X_text = encode_texts(subject_text, item_text, benchmark_text)
    extra = build_extra_for_one(input, U, V, B)

    with torch.no_grad():
        logit = MODEL(X_text, extra)
        p_model = torch.sigmoid(logit).item()

    subject_prior = SUBJECT_MEAN.get(subject_text, GLOBAL_MEAN)
    item_prior = ITEM_MEAN.get(item_text, GLOBAL_MEAN)

    bc = SO_PRIORS["benchmark_condition_prior"].get(
        f"{benchmark_text}|||{condition}", GLOBAL_MEAN
    )

    sb = SO_PRIORS["subject_benchmark_prior"].get(
        f"{subject_text}|||{benchmark_text}", GLOBAL_MEAN
    )

    sc = SO_PRIORS["subject_condition_prior"].get(
        f"{subject_text}|||{condition}", GLOBAL_MEAN
    )

    # optimal mixing
    #p = 0.65 * p_model + 0.35 * prior_score +0*subject_prior
    #p = 0.55 * p_model + 0.25 * prior_score + 0.2 * subject_prior
    #p = 0.60 * p_model + 0.30 * item_prior + 0.10 * subject_prior
    #for stability
    p = 0.60 * p_model + 0.30 * item_prior + 0.10 * subject_prior
    p = clip_prob(p)
    return float(p)