"""
Timbre Text-to-Emotion — Gradio Space
https://huggingface.co/spaces/WCA0202/Timbre-Text-to-Emotion

Single endpoint: classify_mood
  Input:  free-text mood description (string)
  Output: JSON string with emotion label, confidence, and acoustic profile

Called from the Timbre web frontend during onboarding when a client
types a mood description instead of uploading a reference track.
"""

import json
import gradio as gr

from emotion_classifier import classify_text

print("Loading emotion classifier (j-hartmann/emotion-english-distilroberta-base)...")
# Warm up the model on startup so the first real request is fast
classify_text("warm and hopeful")
print("Classifier ready.")


def classify_mood(text: str) -> str:
    """
    Classify a mood description into a Timbre emotion and acoustic profile.

    Returns JSON:
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

    On error, returns JSON: { "error": "<message>" }
    """
    if not text or not text.strip():
        return json.dumps({"error": "No text provided."})

    try:
        result = classify_text(text.strip())
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})


demo = gr.Interface(
    fn=classify_mood,
    inputs=gr.Textbox(
        label="Mood description",
        placeholder="e.g. driving alone at night, feeling weightless and free",
        lines=3,
    ),
    outputs=gr.Textbox(label="Emotion JSON"),
    title="Timbre Text-to-Emotion",
    description=(
        "Classify a free-text mood description into a Timbre emotion category "
        "and return a target acoustic profile (valence, arousal, BPM, mood scores). "
        "Used internally by the Timbre brief engine."
    ),
    api_name="classify_mood",
    examples=[
        ["driving alone at night with the windows down"],
        ["warm and hopeful, like the first day of spring"],
        ["raw and angry, ready to fight back"],
        ["delicate and nostalgic, like an old photo"],
        ["euphoric on the dance floor, nothing matters"],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
