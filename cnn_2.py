import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset_path = "./selected/total_not_selected/depth"
test_file_path = "./selected/captures"
IMAGE_SIZE = (64, 64)

def load_images(base_path):
    data, labels = [], []
    for category in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, category)
        if category.startswith(".") or not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.startswith("."):
                continue
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            data.append(image)
            labels.append(category)
    return np.array(data), np.array(labels)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading training data...")
X_train_raw, train_labels = load_images(dataset_path)

print("Loading test data...")
X_test_raw, test_labels = load_images(test_file_path)

# ==========================================
# 2. PREPROCESS
# ==========================================
X_train = X_train_raw.reshape(-1, 64, 64, 1).astype("float32") / 255.0
X_test  = X_test_raw.reshape(-1, 64, 64, 1).astype("float32") / 255.0

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test  = le.transform(test_labels)

num_classes = len(le.classes_)
print(f"Classes ({num_classes}):", le.classes_)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==========================================
# 3. AUGMENTATION (key fix for domain gap)
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.3),        # simulates lighting differences
    layers.GaussianNoise(0.05),        # simulates sensor noise
], name="augmentation")

# ==========================================
# 4. BUILD CNN WITH BATCH NORM
# ==========================================
inputs = Input(shape=(64, 64, 1))
x = data_augmentation(inputs)          # augment only during training

# Block 1
x = layers.Conv2D(32, (3,3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2,2))(x)

# Block 2
x = layers.Conv2D(64, (3,3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2,2))(x)

# Block 3
x = layers.Conv2D(128, (3,3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2,2))(x)

# Block 4 (extra depth helps with 26 classes)
x = layers.Conv2D(256, (3,3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

# ==========================================
# 5. TRAIN
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# Use test set as validation so you can actually see real performance
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),   # <-- real val, not a split of train
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 6. EVALUATE
# ==========================================
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print("CNN Accuracy:", acc)
print("="*40)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))