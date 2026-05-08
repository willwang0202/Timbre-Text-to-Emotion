"""
Text → Timbre emotion category mapper.

Uses j-hartmann/emotion-english-distilroberta-base (fine-tuned on 6 emotion labels)
to classify the input text, then maps it to one of Timbre's 16 acoustic emotion buckets
and returns a target acoustic profile.
"""

from __future__ import annotations
from transformers import pipeline
import numpy as np

_classifier = None


def _load():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,  # CPU — HF free tier has no GPU
        )
    return _classifier


# Hartmann model labels → Timbre emotion keys
# Model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_HARTMANN_TO_TIMBRE: dict[str, list[tuple[str, float]]] = {
    "joy":      [("euphoric", 0.4), ("uplifting", 0.35), ("passionate", 0.25)],
    "sadness":  [("melancholic", 0.5), ("bittersweet", 0.3), ("longing", 0.2)],
    "anger":    [("aggressive", 0.5), ("tense", 0.3), ("rebellious", 0.2)],
    "fear":     [("tense", 0.5), ("anxious", 0.3), ("mysterious", 0.2)],
    "disgust":  [("dark", 0.5), ("rebellious", 0.3), ("tense", 0.2)],
    "neutral":  [("calm", 0.4), ("nostalgic", 0.35), ("dreamy", 0.25)],
    "surprise": [("euphoric", 0.35), ("uplifting", 0.35), ("dreamy", 0.3)],
}

# Acoustic target profiles per Timbre emotion (valence 1-9, arousal 1-9, mood scores 0-1)
# Derived from dataset medians in recommend_v2.py MOOD_PROFILES.
EMOTION_PROFILES: dict[str, dict] = {
    "euphoric":    {"valence": 7.5, "arousal": 7.2, "bpm": 128, "mood_happy": 0.82, "mood_sad": 0.04, "mood_aggressive": 0.12, "mood_relaxed": 0.18, "mood_party": 0.75, "danceability": 0.80},
    "uplifting":   {"valence": 7.0, "arousal": 6.2, "bpm": 118, "mood_happy": 0.72, "mood_sad": 0.06, "mood_aggressive": 0.08, "mood_relaxed": 0.35, "mood_party": 0.50, "danceability": 0.65},
    "passionate":  {"valence": 6.2, "arousal": 6.8, "bpm": 110, "mood_happy": 0.55, "mood_sad": 0.18, "mood_aggressive": 0.38, "mood_relaxed": 0.22, "mood_party": 0.35, "danceability": 0.55},
    "melancholic": {"valence": 3.5, "arousal": 3.2, "bpm": 72,  "mood_happy": 0.08, "mood_sad": 0.72, "mood_aggressive": 0.05, "mood_relaxed": 0.55, "mood_party": 0.03, "danceability": 0.20},
    "bittersweet": {"valence": 4.8, "arousal": 3.8, "bpm": 82,  "mood_happy": 0.30, "mood_sad": 0.52, "mood_aggressive": 0.08, "mood_relaxed": 0.45, "mood_party": 0.10, "danceability": 0.30},
    "longing":     {"valence": 3.8, "arousal": 3.5, "bpm": 76,  "mood_happy": 0.12, "mood_sad": 0.62, "mood_aggressive": 0.06, "mood_relaxed": 0.50, "mood_party": 0.05, "danceability": 0.22},
    "aggressive":  {"valence": 3.2, "arousal": 7.8, "bpm": 148, "mood_happy": 0.10, "mood_sad": 0.12, "mood_aggressive": 0.85, "mood_relaxed": 0.05, "mood_party": 0.30, "danceability": 0.55},
    "tense":       {"valence": 3.0, "arousal": 6.5, "bpm": 130, "mood_happy": 0.08, "mood_sad": 0.25, "mood_aggressive": 0.60, "mood_relaxed": 0.08, "mood_party": 0.15, "danceability": 0.40},
    "rebellious":  {"valence": 4.0, "arousal": 7.2, "bpm": 138, "mood_happy": 0.18, "mood_sad": 0.15, "mood_aggressive": 0.72, "mood_relaxed": 0.10, "mood_party": 0.38, "danceability": 0.58},
    "anxious":     {"valence": 3.2, "arousal": 6.0, "bpm": 115, "mood_happy": 0.08, "mood_sad": 0.35, "mood_aggressive": 0.45, "mood_relaxed": 0.12, "mood_party": 0.10, "danceability": 0.35},
    "mysterious":  {"valence": 4.2, "arousal": 4.5, "bpm": 88,  "mood_happy": 0.15, "mood_sad": 0.38, "mood_aggressive": 0.28, "mood_relaxed": 0.38, "mood_party": 0.12, "danceability": 0.38},
    "dark":        {"valence": 2.8, "arousal": 5.2, "bpm": 98,  "mood_happy": 0.05, "mood_sad": 0.55, "mood_aggressive": 0.55, "mood_relaxed": 0.12, "mood_party": 0.08, "danceability": 0.35},
    "calm":        {"valence": 5.8, "arousal": 2.8, "bpm": 68,  "mood_happy": 0.38, "mood_sad": 0.22, "mood_aggressive": 0.04, "mood_relaxed": 0.82, "mood_party": 0.08, "danceability": 0.25},
    "nostalgic":   {"valence": 5.2, "arousal": 3.5, "bpm": 78,  "mood_happy": 0.35, "mood_sad": 0.42, "mood_aggressive": 0.06, "mood_relaxed": 0.58, "mood_party": 0.10, "danceability": 0.30},
    "dreamy":      {"valence": 5.5, "arousal": 3.0, "bpm": 72,  "mood_happy": 0.40, "mood_sad": 0.30, "mood_aggressive": 0.05, "mood_relaxed": 0.72, "mood_party": 0.08, "danceability": 0.28},
    "romantic":    {"valence": 6.5, "arousal": 4.2, "bpm": 88,  "mood_happy": 0.55, "mood_sad": 0.28, "mood_aggressive": 0.06, "mood_relaxed": 0.60, "mood_party": 0.18, "danceability": 0.42},
}


def classify_text(text: str) -> dict:
    """
    Classify mood text into a Timbre emotion + acoustic profile.

    Returns:
        {
            "emotion": "melancholic",
            "confidence": 0.87,
            "acoustic_profile": { "valence": 3.5, "arousal": 3.2, ... }
        }
    """
    clf = _load()
    raw: list[dict] = clf(text)[0]  # list of {label, score}

    # Build weighted Timbre emotion scores
    timbre_scores: dict[str, float] = {}
    for item in raw:
        label = item["label"]
        score = item["score"]
        for timbre_emotion, weight in _HARTMANN_TO_TIMBRE.get(label, []):
            timbre_scores[timbre_emotion] = timbre_scores.get(timbre_emotion, 0.0) + score * weight

    if not timbre_scores:
        timbre_scores["calm"] = 1.0

    # Top emotion
    best_emotion = max(timbre_scores, key=timbre_scores.__getitem__)
    total = sum(timbre_scores.values())
    confidence = round(timbre_scores[best_emotion] / total, 3) if total > 0 else 0.0

    profile = EMOTION_PROFILES.get(best_emotion, EMOTION_PROFILES["calm"])

    return {
        "emotion": best_emotion,
        "confidence": confidence,
        "acoustic_profile": profile,
    }
