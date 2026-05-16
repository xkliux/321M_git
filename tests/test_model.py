import importlib
import math
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission"
sys.path.insert(0, str(SUBMISSION_DIR))
model = importlib.import_module("model")


def test_sample_input():
    x = {
        "benchmark": "MMLU",
        "condition": "none",
        "subject_content": "GPT-4 OpenAI large language model",
        "item_content": "What is 2 + 2?",
    }

    p = model.predict(x)

    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_output_real_number():
    x = {
        "benchmark": "test",
        "condition": "none",
        "subject_content": "",
        "item_content": "",
    }

    p = model.predict(x)

    assert math.isfinite(p)


def test_incomplete_input():
    x = {}

    p = model.predict(x)

    assert isinstance(p, float)
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_clipping():
    assert model.clip_prob(-100.0) == 0.05
    assert model.clip_prob(100.0) == 0.95
    assert model.clip_prob(0.5) == 0.5
