import os
from predict import predict  # Make sure predict() returns the label (not just prints)

test_dir = "data/test"
total = 0
correct = 0

for video_file in os.listdir(test_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(test_dir, video_file)
        
        print(f"\n--- Predicting: {video_path} ---")
        prediction = predict(video_path, return_label=True)  # return_label should be handled in predict.py

        # Use file naming rule to determine actual label
        actual = "Real" if "_" in video_file else "Fake"

        print(f"Actual: {actual}, Predicted: {prediction}")

        if prediction == actual:
            correct += 1
        total += 1

# Calculate and print accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")
else:
    print("No test videos found.")
