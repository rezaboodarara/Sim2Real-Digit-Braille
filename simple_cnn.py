import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

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

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {img_path}")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        data.append(image)
        labels.append(category)

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

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {img_path}")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        test_data.append(image)
        test_labels.append(category)

# ==========================================
# 3. PREPROCESS & LABEL ENCODE
# ==========================================
# Reshape for CNN: (samples, height, width, channels)
X_train = np.array(data).reshape(-1, 64, 64, 1).astype("float32") / 255.0
X_test  = np.array(test_data).reshape(-1, 64, 64, 1).astype("float32") / 255.0

le = LabelEncoder()
y_train = le.fit_transform(labels)
y_test  = le.transform(test_labels)

num_classes = len(le.classes_)
print(f"Classes ({num_classes}):", le.classes_)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==========================================
# 4. BUILD SIMPLE CNN
# ==========================================
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Classifier head
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.summary()

# ==========================================
# 5. TRAIN
# ==========================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,     # 10% of train used for val
    verbose=1
)

# ==========================================
# 6. EVALUATE
# ==========================================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\n" + "="*40)
print("CNN Accuracy:", acc)
print("="*40)
print(classification_report(y_test, y_pred, target_names=le.classes_))