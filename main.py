from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import torch.nn.functional as F

app = FastAPI()

MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

PROMPTS = [
    "a photo of a national ID card",
    "a photo of a passport",
    "a photo of a driver's license",
    "a random photo",
    "a selfie",
    "a landscape"
]

ID_PROMPTS = PROMPTS[:3]
THRESHOLD = 0.28


@app.post("/verify-id")
async def verify_id(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    inputs = processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = F.cosine_similarity(text_features, image_features)

    scores = {p: float(s) for p, s in zip(PROMPTS, similarities)}

    best_id_score = max(scores[p] for p in ID_PROMPTS)

    return {
        "is_id_card": best_id_score >= THRESHOLD,
        "score": best_id_score,
        "scores": scores
    }
