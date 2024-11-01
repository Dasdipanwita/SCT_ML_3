import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare and save training data
data_dir = "E:/cat vs dog/train"  # Update to the path where training images are stored
categories = ["cat", "dog"]
data = []

for img in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img)
    pet_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    pet_img = cv2.resize(pet_img, (50, 50)).flatten()  # Resize and flatten the image
    label = 0 if img.startswith("cat") else 1  # Labeling based on file name
    data.append([pet_img, label])

# Save data to a pickle file
with open("final_data.pickle", "wb") as pick_out:
    pickle.dump(data, pick_out)

print("Data prepared and saved to final_data.pickle")

# Step 2: Load data from the pickle file for training
with open("final_data.pickle", "rb") as pick_in:
    data = pickle.load(pick_in)

# Separate features and labels
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Step 3: Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 5: Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Step 6: Validate model on validation data
y_val_pred = svm_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred) * 100)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred, target_names=["cat", "dog"]))
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# Step 7: Load and preprocess test data for final predictions
test_dir = "E:/cat vs dog/test1"  # Directory with test images
test_images = []
test_filenames = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    pet_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pet_img = cv2.resize(pet_img, (50, 50)).flatten()
    test_images.append(pet_img)
    test_filenames.append(img_name)

# Standardize test images
X_test = scaler.transform(np.array(test_images))

# Step 8: Predict on test images
y_test_pred = svm_model.predict(X_test)

# Output predictions as "cat" or "dog"
predictions = ["cat" if label == 0 else "dog" for label in y_test_pred]
for img_name, pred in zip(test_filenames, predictions):
    print(f"Image: {img_name} - Prediction: {pred}")
