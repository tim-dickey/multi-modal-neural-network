"""Example inference script for multi-modal neural network."""

# Ensure project's `src` is importable when running as a script

# Project imports are done inside `load_model` to avoid manipulating sys.path
# at module import time
import argparse

import torch
from PIL import Image
from torchvision import transforms

# Project imports are performed inside `load_model` to avoid module-level
# side-effects when running linters or importing this module.


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    # Ensure `src` is importable when this script is executed as a script
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent))

    from src.models import create_multi_modal_model
    from src.utils.config import load_config

    config = load_config(config_path)
    model = create_multi_modal_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def preprocess_image(image_path: str, img_size: int = 224):
    """Preprocess image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image


def tokenize_text(text: str, max_length: int = 512):
    """Simple tokenization (replace with proper tokenizer in production)."""
    # This is a placeholder - use transformers.AutoTokenizer in production
    input_ids = [1] + [min(ord(c) % 30000, 30521) for c in text[: max_length - 2]] + [2]
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
        input_ids.append(0)
        attention_mask.append(0)

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with multi-modal model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, default="", help="Optional text input")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.config, args.checkpoint, args.device)

    # Preprocess inputs
    print(f"Processing image: {args.image}")
    image = preprocess_image(
        args.image,
        img_size=config.get("model", {}).get("vision_encoder", {}).get("img_size", 224),
    )
    image = image.to(args.device)

    if args.text:
        print(f"Processing text: {args.text}")
        text_encoding = tokenize_text(args.text)
        input_ids = text_encoding["input_ids"].to(args.device)
        attention_mask = text_encoding["attention_mask"].to(args.device)
    else:
        input_ids = None
        attention_mask = None

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(
            images=image, input_ids=input_ids, attention_mask=attention_mask
        )

    logits = outputs["logits"]
    predictions = logits.argmax(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)

    # Print results
    print("\nResults:")
    print(f"  Predicted class: {predictions[0].item()}")
    print(f"  Confidence: {probabilities[0, predictions[0]].item():.4f}")
    print("\nTop 5 predictions:")
    top5_probs, top5_indices = torch.topk(probabilities[0], k=5)
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"  {i + 1}. Class {idx.item()}: {prob.item():.4f}")


if __name__ == "__main__":
    main()
