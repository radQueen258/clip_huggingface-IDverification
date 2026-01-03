from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

model = None
processor = None

def get_model():
    global model, processor
    if model is None:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            torch_dtype=torch.float32
        )
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        model.eval()
    return model, processor


@app.post("/verify-id")
async def verify_id(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    model, processor = get_model()

    texts = ["an ID card", "a random photo"]

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    probability = probs[0].item()

    return {
        "accepted": probability > 0.6,
        "probability": probability,
        "method": "clip"
    }
