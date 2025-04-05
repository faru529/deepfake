import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from dataset import DeepfakeVideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter
import os
# Set dataset path as a string
dataset_root = "c:/Users/FARHAT AYESHA/Desktop/final p/DeepfakeDetection/data/FakeAVCeleb"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pass the correct dataset root (string)
dataset = DeepfakeVideoDataset(root_dir=dataset_root, transform=transform)
print("Number of videos found:", len(dataset))
print("Sample paths:", dataset.video_paths[:5])  # Print first 5 paths
# Count the number of real (0) and fake (1) labels
label_counts = Counter(dataset.labels)  # Assuming your dataset class has a 'labels' attribute

print("Class distribution in dataset:")
print(f"Real videos (label=0): {label_counts[0]}")
print(f"Fake videos (label=1): {label_counts[1]}")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Test a batch
for frames, labels in dataloader:
    print(f"Batch of frames shape: {frames.shape}")  # Expected: (batch_size, num_frames, 3, 224, 224)
    print(f"Labels: {labels}")
    break

# Load EfficientNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 2)  # Binary Classification (Real vs Fake)
model.to(device)

#for frames, labels in dataloader:
 #   frames = frames.to(device)  # Move batch of frames to GPU
  #  labels = labels.to(device)  # Move batch of labels to GPU
for frames, labels in dataloader:
    frames = frames[:, 0, :, :, :].to(device)  # Select only the first frame (shape: [batch_size, 3, 224, 224])
    labels = labels.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model ready for training on", device)
from tqdm import tqdm  # optional for progress bar

# Training configuration
num_epochs = 5  # you can increase later
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(dataloader):
        frames = frames.to(device)
        labels = labels.to(device)

        # Flatten batch: (B, T, 3, 224, 224) -> (B*T, 3, 224, 224)
        frames = frames.view(-1, 3, 224, 224)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels.repeat_interleave(frames.shape[0] // labels.shape[0]))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.repeat_interleave(frames.shape[0] // labels.shape[0])).sum().item()
        total += labels.shape[0] * (frames.shape[0] // labels.shape[0])

    epoch_loss = running_loss / len(dataloader)

    accuracy = correct / total * 100

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
   
torch.save(model.state_dict(), "models/efficientnet_model.pth")

print("Model saved successfully!")
