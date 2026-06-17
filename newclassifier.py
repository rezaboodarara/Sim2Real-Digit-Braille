import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configurable parameters
dataset_path = "./selected/total_not_selected/depth"
categories = os.listdir(dataset_path)
IMAGE_SIZE = (64, 64)

data = []
labels = []

# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if category.startswith("."):
        continue
    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue
        img_path = os.path.join(folder_path, filename)
        
        # Load image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        
        # Resize
        image = cv2.resize(image, IMAGE_SIZE)
        
        # Flatten image into 1D vector (Your original logic)
        flat = image.flatten()
        data.append(flat)
        labels.append(category)

X = np.array(data)
y = np.array(labels)

print(f"Shape of X (Train): {X.shape}")
print(f"Shape of y (Train): {y.shape}")

# Use ALL of your current data for training
X_train = X

# ==========================================
# 2. LOAD TESTING DATA
# ==========================================
test_file_path = "./selected/captures" 
test_categories = os.listdir(test_file_path)

test_data = []
test_labels = []

for category in test_categories:
    folder_path = os.path.join(test_file_path, category)
    if category.startswith(".") or not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue
        img_path = os.path.join(folder_path, filename)

        # Fixed: Match the training preprocessing (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to load image {img_path}")
            continue

        # Resize (Using the same IMAGE_SIZE = (64,64) as training)
        image = cv2.resize(image, IMAGE_SIZE)

        # Flatten and store
        flat = image.flatten()
        test_data.append(flat)
        test_labels.append(category)

# Assign to X_test
X_test = np.array(test_data) 

print(f"Shape of X_test: {X_test.shape}")
print("Testing samples loaded:", len(X_test))

# ==========================================
# 3. LABEL ENCODING
# ==========================================
le = LabelEncoder()
y_train = le.fit_transform(labels)       # FIT and transform on training data
y_test = le.transform(test_labels)       # ONLY transform on testing data

print("Classes:", le.classes_)

# ==========================================
# 4. DECISION TREE TRAINING & EVALUATION
# ==========================================
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\n" + "="*40)
print("Decision Tree Accuracy:", acc)
print("="*40)

# Detailed report
print(classification_report(y_test, y_pred, target_names=le.classes_))
