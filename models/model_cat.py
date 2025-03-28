import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV3Small

IMG_SIZE = 144

# Load pretrained teacher model (imagenet)
teacher_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.75,
    include_top=False,  # Custom classification head
    weights='imagenet',
    pooling='avg'
)
teacher_model.trainable = False

# Teacher classifier head (binary)
teacher_classifier = tf.keras.Sequential([
    teacher_model,
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Smaller student model (alpha=0.2), no pretrained weights
student_base = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.2,
    include_top=False,
    weights=None,
    pooling='avg'
)
student_classifier = tf.keras.Sequential([
    student_base,
    tf.keras.layers.Dense(2, activation='sigmoid')
])

import tensorflow_datasets as tfds

# Load the dataset.
raw_ds = tfds.load('cats_vs_dogs', split='train', as_supervised=True)

# Define the cat label(s) with matching dtype.
cats = tf.constant([0], dtype=tf.int64)

# Function to assign new labels: 0 if cat, 1 otherwise.
def assign_new_label(image, label):
    # Check if the label is a cat.
    is_cat = tf.reduce_any(tf.equal(label, cats))
    new_label = tf.cond(is_cat, lambda: tf.constant(0, dtype=tf.int64),
                              lambda: tf.constant(1, dtype=tf.int64))
    return image, new_label

# Map the new labels.
dataset = raw_ds.map(assign_new_label)

# Preprocess images.
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

dataset = dataset.map(preprocess)

import tensorflow as tf
from tensorflow.keras.datasets import cifar100

(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar100.load_data(label_mode='fine')

# Preprocess function to match your existing dataset
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply binary label mapping and preprocessing to CIFAR
cat_class = 3

def assign_and_preprocess(image, label):
    binary_label = tf.cond(tf.equal(label, cat_class),
                           lambda: tf.constant(0, dtype=tf.int64),
                           lambda: tf.constant(1, dtype=tf.int64))
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, binary_label

# Combine train and test splits
images_cifar = tf.concat([x_train_cifar, x_test_cifar], axis=0)
labels_cifar = tf.concat([y_train_cifar, y_test_cifar], axis=0)


# Apply mapping
cifar_ds = tf.data.Dataset.from_tensor_slices((images_cifar, labels_cifar))
cifar_ds = cifar_ds.map(assign_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Combine the existing dataset with CIFAR dataset
combined_ds = dataset.concatenate(cifar_ds).shuffle(80000)

# Shuffle combined dataset
combined_ds = combined_ds.shuffle(buffer_size=80000)

# Compute combined dataset size
combined_size = tf.data.experimental.cardinality(combined_ds).numpy()
train_size = int(0.8 * combined_size)

# Final Train and Validation split
train_ds = combined_ds.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
validation_ds = combined_ds.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

def preprocess(images, labels):
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    labels = 1 - tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))
    return images, labels

train_ds_prep = train_ds.map(preprocess)
val_ds_prep = validation_ds.map(preprocess)

# import matplotlib.pyplot as plt
# 
# for images, labels in train_ds_prep.take(1):
#     plt.figure(figsize=(8, 8))
#     for i in range(9):
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.title("Cat" if labels.numpy()[i][0] == 1 else "Not Cat")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
# 
# for images, labels in train_ds_prep.take(1):
#     preds = teacher_classifier(images, training=False)
#     print("Predictions (before training):", preds.numpy().flatten()[:10])

from collections import Counter

# Use your training label distribution
train_labels = []
for _, label in train_ds.unbatch():
    train_labels.append(int(label.numpy()))

counts = Counter(train_labels)
total = sum(counts.values())
class_weight = {
    0: total / (2 * counts[0]),  # weight for class 0 (cats)
    1: total / (2 * counts[1])   # weight for class 1 (non-cats)
}

print("Class weights:", class_weight)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      # or 'val_accuracy'
    patience=5,              # wait for 5 epochs with no improvement
    restore_best_weights=True,  # restores weights from best epoch
)

def build_and_train_teacher_model():
    teacher_classifier.compile(
        optimizer=Adam(0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    teacher_classifier.fit(
        train_ds_prep,
        validation_data=val_ds_prep,
        epochs=100,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    teacher_classifier.trainable = False

    return teacher_classifier

import os
from tensorflow.keras.models import load_model

if os.path.exists("teacher_model.keras"):
    print("✅ Loaded pre-trained teacher.")
    teacher_classifier = load_model("teacher_model.keras", compile=False)
else:
    print("🚧 Training teacher model...")
    teacher_classifier = build_and_train_teacher_model()
    teacher_classifier.save("teacher_model.keras")

for batch in train_ds_prep.take(1):
    print("Batch type:", type(batch))
    if isinstance(batch, tuple):
        print("Batch length:", len(batch))
        for i, element in enumerate(batch):
            print(f"Element {i} shape: {element.shape}, dtype: {element.dtype}")
    else:
        print("Not a tuple:", batch)

from keras.saving import register_keras_serializable

@register_keras_serializable()
class DistillationModel(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.7):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.kld = tf.keras.losses.KLDivergence()

    def compile(self, optimizer, loss_fn=None, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.loss_fn = loss_fn or tf.keras.losses.BinaryCrossentropy()


    def distillation_loss(self, y_true, teacher_pred, student_pred):
        hard_loss = self.loss_fn(y_true, student_pred)

        # Soften predictions for teacher and student
        t = self.temperature
        teacher_soft = tf.nn.sigmoid(teacher_pred / t)
        student_soft = tf.nn.sigmoid(student_pred / t)

        soft_loss = tf.keras.losses.binary_crossentropy(teacher_soft, student_soft)
        soft_loss = tf.reduce_mean(soft_loss)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


    def train_step(self, data):
        images, labels = data[:2]
        teacher_preds = self.teacher(images, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(images, training=True)

            # Soften
            t = self.temperature
            teacher_soft = tf.nn.sigmoid(teacher_preds / t)
            student_soft = tf.nn.sigmoid(student_preds / t)

            soft_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(teacher_soft, student_soft))
            hard_loss = self.loss_fn(labels, student_preds)
            loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # ✅ Keras-managed metrics
        self.compiled_metrics.update_state(labels, student_preds)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data[:2]
        preds = self.student(images, training=False)
        loss = self.loss_fn(labels, preds)

        self.compiled_metrics.update_state(labels, preds)
        return {"val_loss": loss, **{m.name: m.result() for m in self.metrics}}

early_stop_distil = EarlyStopping(
    monitor='val_val_loss',      # or 'val_accuracy'
    patience=5,              # wait for 5 epochs with no improvement
    restore_best_weights=True,  # restores weights from best epoch
    mode='min'
)

distill_model = DistillationModel(student=student_classifier, teacher=teacher_classifier)

distill_model.compile(
    optimizer=Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

if os.path.exists("distill_model.keras"):
    print("✅ Loaded pre-trained distill_model.")
    distill_model = load_model("distill_model.keras", compile=False)
else:
  # Train with fit() - simple and clean
  distill_model.fit(
      train_ds_prep,
      validation_data=val_ds_prep,
      epochs=100,
      class_weight=class_weight,
      callbacks=[early_stop_distil]
  )

distill_model.save("distill_model.keras")

student_model = distill_model.student
student_model.save("student_model.keras")

def representative_data_gen():
    for images, _ in train_ds_prep.take(100):
        yield [images]

# Load model if needed
model = tf.keras.models.load_model("student_model.keras")

# Converter setup
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable dynamic range quantization (weights only, safest)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save
with open("student_model_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as student_model_quant.tflite")

# Print size
size_kb = os.path.getsize("student_model_quant.tflite") / 1024
print(f"✅ TFLite model saved: student_model_quant.tflite ({size_kb:.2f} KB)")

import tensorflow as tf
import numpy as np
from pathlib import Path

# Load your quantized TFLite model
interpreter = tf.lite.Interpreter(model_path="student_model_quant.tflite")
interpreter.allocate_tensors()

# Get input quantization parameters
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scale, zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']
input_shape = input_details[0]['shape']  # [1, height, width, 3]
input_height, input_width, input_channels = input_shape[1:]

print(f"Input scale: {scale}, zero point: {zero_point}")
print(f"Model expects input shape: {input_shape}")

# Helper: Convert flattened int8 array to C-style string
def to_c_array(data, var_name):
    c_array = f"const int8_t {var_name}[] = {{\n    "
    values = [str(x) for x in data]
    for i in range(0, len(values), 12):
        c_array += ", ".join(values[i:i+12]) + ",\n    "
    c_array = c_array.rstrip(",\n    ") + "\n};\n"
    return c_array

# Set number of images to export
N = 5
c_arrays = ""
input_data_list = []

unbatched_ds = train_ds_prep.unbatch()

# Loop over dataset
for i, (image, label) in enumerate(unbatched_ds.take(N)):
    image = tf.squeeze(image)
    print("Pixel min:", image.numpy().min())
    print("Pixel max:", image.numpy().max())
    print("Pixel mean:", image.numpy().mean())

    # 🔧 Resize to match model input shape
    image = tf.image.resize(image, (input_height, input_width))

    # Normalize image from [0.0, 1.0] to [-1.0, 1.0]
    image_centered = (image.numpy() - 0.5) * 2.0

    # Quantize to int8 [-128, 127]
    image_int8 = np.clip(image_centered * 127, -128, 127).astype(np.int8)
    print("Python int8 sample:", image_int8.flatten()[:20])
    flattened = image_int8.flatten()
    input_data_list.append(flattened.copy())
    print(f"Image {i} resized shape: {image.shape}, flattened length: {len(flattened)}")

    # Determine label name
    label_val = label.numpy()
    label_name = "Cat" if int(label_val[0]) == 1 else "Not Cat"

    # Append to header
    c_arrays += f"// Image {i} - Label: {label_name}\n"
    c_arrays += to_c_array(flattened, f"input_data_{i}") + "\n"

# Create header file
header = "#ifndef IMAGES_H\n#define IMAGES_H\n\n"
footer = "#endif // IMAGES_H\n"
full_content = header + c_arrays + footer

# Save to file in Colab
Path("images.h").write_text(full_content)
print("✅ Saved images.h in current Colab directory.")

print("Input dtype:", input_details[0]['dtype'])  # Should print tf.uint8 or tf.int8

print(144*144*3)

for i, flattened in enumerate(input_data_list):
    input_array = np.array(flattened, dtype=np.int8).reshape((1, input_height, input_width, input_channels))

    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    output_q = interpreter.get_tensor(output_details[0]['index'])[0]
    output_f = (output_q.astype(np.float32) - output_zero_point) * output_scale
    probs = tf.nn.softmax(output_f).numpy()
    # prediction = "Cat" if probs[0] > probs[1] else "Not Cat"

    print(f"Image {i}:")
    print(f"  Raw int8 output: {output_q}")
    print(f"  Dequantized: {output_f}")
    print(f"  Softmax: {probs}")
    # print(f"  ✅ Predicted: {prediction}\n")
