import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=5):
        self.video_paths = []
        self.labels = []

        # Scan all subdirectories
        for category, label in [("RealVideo-RealAudio", 0), ("FakeVideo-RealAudio", 1)]:
            category_path = os.path.join(root_dir, category)
            for root, _, files in os.walk(category_path):  # Traverse all subfolders
                for file in files:
                    if file.endswith(".mp4"):
                        self.video_paths.append(os.path.join(root, file))
                        self.labels.append(label)

        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def extract_random_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return None

        frame_indices = np.linspace(0, frame_count-1, self.num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.extract_random_frames(video_path)

        if frames is None:
            return self.__getitem__((idx + 1) % len(self.video_paths))  # Handle empty video

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)
