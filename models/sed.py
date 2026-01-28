import torch
from panns_inference import AudioTagging

SAMPLE_RATE = 32000
WINDOW_MS = 1000
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioTagging(
    checkpoint_path="models/panns_data/Cnn14_mAP=0.431.pth",
    device=device
)

labels = model.labels


TARGET_LABELS = [
    "Gunshot, gunfire",
    "Explosion",
    "Glass",
    "Breaking",
    "Thump",
    "Impact",
    "Slap, smack",
    "Whack, thwack"
]


def detect_impacts(audio_300ms, top_k=10):
    audio_batch = audio_300ms[None, :]

    with torch.no_grad():
        scores = model.inference(audio_batch)[0].reshape(-1)

    label_scores = [(label, float(score)) for label, score in zip(labels, scores)]

    label_scores.sort(key=lambda x: x[1], reverse=True)

    filtered = [
        (label, score)
        for label, score in label_scores
        if any(t.lower() in label.lower() for t in TARGET_LABELS)
    ]

    return filtered[:top_k], label_scores[:top_k]