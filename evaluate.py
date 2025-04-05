import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from predict import predict

# Path to folder containing test videos
TEST_FOLDER = "data/test_new"

# Store actual vs predicted
actual_labels = []
predicted_labels = []
confidence_scores = []

# Label mapping
label_to_int = {"Real": 0, "Fake": 1}
int_to_label = {0: "Real", 1: "Fake"}

print("\nEvaluating...\n")

# Loop through test videos
for video_file in os.listdir(TEST_FOLDER):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(TEST_FOLDER, video_file)
    actual_label = "Real" if "_" in video_file else "Fake"

    # Predict returns predicted label AND confidence score
    predicted_label, confidence = predict(video_path, return_label=True, return_confidence=True)

    actual_labels.append(label_to_int[actual_label])
    predicted_labels.append(label_to_int[predicted_label])
    confidence_scores.append(confidence)

    correct = predicted_label == actual_label
    print(f"{video_file:<30} | Actual: {actual_label:<5} | Predicted: {predicted_label:<5} | Confidence: {confidence:.2f} | {'✔' if correct else '✘'}")

# Accuracy
accuracy = sum([1 for a, p in zip(actual_labels, predicted_labels) if a == p]) / len(actual_labels) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(actual_labels, predicted_labels, target_names=["Real", "Fake"]))

# Confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
