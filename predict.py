import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
import sys

# Load model
MODEL_PATH = "models/efficientnet_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = EfficientNet.from_name("efficientnet-b0", num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
    cap.release()
    return torch.stack(frames) if frames else None

def predict(video_path, return_label=False, return_confidence=False):
    model = load_model()
    frames = extract_frames(video_path)

    if frames is None:
        print("Error: Could not extract frames.")
        return None if not return_label else ("Unknown", 0.0)

    frames = frames.to(DEVICE)

    with torch.no_grad():
        outputs = model(frames)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        predicted_class = torch.argmax(avg_probs).item()
        confidence = avg_probs[predicted_class].item()

    label = "Fake" if predicted_class == 1 else "Real"

    if return_label and return_confidence:
        return label, confidence
    elif return_label:
        return label
    else:
        print(f"Prediction: {label} (Confidence: {confidence:.2f})")

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <video_path>")
    else:
        predict(sys.argv[1])
