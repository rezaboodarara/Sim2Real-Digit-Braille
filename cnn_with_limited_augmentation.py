import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================================
# CONFIGURABLE PARAMETERS
# ==========================================
dataset_path = "./selected/total_not_selected/depth"   # simulator data (train + val)
test_file_path = "./selected/captures"                  # real data (test only)
IMAGE_SIZE = (64, 64)
VAL_SPLIT = 0.2      # 20% of simulator data used for validation
BATCH_SIZE = 64
EPOCHS = 30
RANDOM_STATE = 42

# ==========================================
# 1. LOAD DATA
# ==========================================
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
                print(f"Failed to load: {img_path}")
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            data.append(image)
            labels.append(category)
    return np.array(data), np.array(labels)

print("Loading simulator (training) data...")
X_sim, sim_labels = load_images(dataset_path)

print("Loading real (test) data...")
X_test_raw, test_labels = load_images(test_file_path)

# ==========================================
# 2. LABEL ENCODING
# ==========================================
le = LabelEncoder()
y_sim = le.fit_transform(sim_labels)
y_test = le.transform(test_labels)

num_classes = len(le.classes_)
print(f"\nClasses ({num_classes}):", le.classes_)

# ==========================================
# 3. TRAIN / VAL SPLIT (simulator data only)
#    Real test data is never touched until final evaluation.
# ==========================================
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_sim, y_sim,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_sim          # ensures all 26 classes are balanced in val
)

# Normalize and reshape for CNN
X_train = X_train_raw.reshape(-1, 64, 64, 1).astype("float32") / 255.0
X_val   = X_val_raw.reshape(-1, 64, 64, 1).astype("float32") / 255.0
X_test  = X_test_raw.reshape(-1, 64, 64, 1).astype("float32") / 255.0

print(f"\nTrain  (simulator): {X_train.shape}")
print(f"Val    (simulator): {X_val.shape}")
print(f"Test   (real data): {X_test.shape}")
print("\n✅ Real test data is isolated — no leakage possible.\n")

# ==========================================
# 4. DATA AUGMENTATION
#    Simulates real-world variation to bridge the sim-to-real gap.
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.3),     # simulates lighting differences
    layers.GaussianNoise(0.05),     # simulates sensor noise
], name="augmentation")

# ==========================================
# 5. BUILD CNN
# ==========================================
inputs = Input(shape=(64, 64, 1))
x = data_augmentation(inputs)          # augmentation only active during training

# Block 1
x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 2
x = layers.Conv2D(64, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 3
x = layers.Conv2D(128, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 4
x = layers.Conv2D(256, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)

# Classifier head
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

# ==========================================
# 6. COMPILE
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 7. TRAIN
#    Validation is done on simulator data only.
#    Real test data is untouched until step 8.
# ==========================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        "best_model_limited.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
]

print("\n" + "="*50)
print("TRAINING (validation on simulator data only)")
print("="*50 + "\n")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),   # simulator val — no leakage
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 8. FINAL EVALUATION ON REAL DATA
#    This is the FIRST and ONLY time the model sees real images.
# ==========================================
print("\n" + "="*50)
print("FINAL EVALUATION ON REAL CAPTURED DATA")
print("(Model sees real data for the first time)")
print("="*50 + "\n")

y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"Final Accuracy on Real Data: {acc:.4f} ({acc*100:.2f}%)")
print("="*50)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))