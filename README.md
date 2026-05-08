---
title: Timbre Text to Emotion
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "5.25.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Timbre Text-to-Emotion

Classifies a free-text mood description into one of Timbre's 16 emotion categories and returns a target acoustic profile. Used by the Timbre brief engine when a client types a mood description during onboarding instead of uploading a reference track.

**Space**: `WCA0202/Timbre-Text-to-Emotion`
**GitHub**: `willwang0202/Timbre-Text-to-Emotion`

---

## API

Single endpoint exposed via the Gradio SSE pattern:

```
POST /call/classify_mood
Body: { "data": ["your mood description here"] }
→ { "event_id": "abc123" }

GET /call/classify_mood/{event_id}
→ SSE stream — read "event: complete" then the next "data:" line
```

### Response

```json
{
  "emotion": "melancholic",
  "confidence": 0.87,
  "acoustic_profile": {
    "valence": 3.5,
    "arousal": 3.2,
    "bpm": 72,
    "mood_happy": 0.08,
    "mood_sad": 0.72,
    "mood_aggressive": 0.05,
    "mood_relaxed": 0.55,
    "mood_party": 0.03,
    "danceability": 0.20
  }
}
```

On error: `{ "error": "<message>" }`

### Emotion taxonomy (16 categories)

`euphoric`, `uplifting`, `passionate`, `melancholic`, `bittersweet`, `longing`, `aggressive`, `tense`, `rebellious`, `anxious`, `mysterious`, `dark`, `calm`, `nostalgic`, `dreamy`, `romantic`

---

## Model

[`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — DistilRoBERTa fine-tuned on the MELD, EmotionLines, ISEAR, WASSA, CrowdFlower, and GoEmotions datasets.

- **Size**: ~82 MB
- **Labels**: anger, disgust, fear, joy, neutral, sadness, surprise
- **Runtime**: CPU-only (HF free tier)

The 7 model labels are mapped to Timbre's 16 emotions via a weighted blend table in `emotion_classifier.py`. Multi-label inputs produce meaningful compound outputs (e.g. a text scoring high on both joy and sadness maps toward `bittersweet`).

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface — `classify_mood` endpoint, model warm-up on startup |
| `emotion_classifier.py` | Loads model, weighted label→emotion mapping, baked-in `EMOTION_PROFILES` |
| `requirements.txt` | `gradio==5.25.0`, `transformers==4.40.2`, `torch==2.2.2`, `numpy` |

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # http://localhost:7860
```

The model (~82 MB) is downloaded from HF Hub on first run and cached in `~/.cache/huggingface/`.

---

## Deploying

Push to both remotes to keep GitHub and HF Space in sync:

```bash
git push origin main   # → GitHub
git push hf main       # → HF Space
```
