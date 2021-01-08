import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

# Data from: Henry Z. Lo. and Cohen, Joseph Paul “Academic Torrents: Scalable Data Distribution.” Neural Information Processing Systems Challenges in Machine Learning (CiML) Workshop, 2016, http://arxiv.org/abs/1603.04395.
# Cohen, Joseph Paul, and Henry Z. Lo. “Academic Torrents: A Community-Maintained Distributed Repository.” Annual Conference of the Extreme Science and Engineering Discovery Environment, 2014, http://doi.org/10.1145/2616498.2616528.

# Load in pre-made numpy arrays of images
file_dir = r"C:\Users\Finn\Manual_Python\Data_Science\Cancer_ML"
inputs_0 = np.load(f"{file_dir}/Class_0_data.npy")
inputs_1 = np.load(f"{file_dir}/Class_1_data.npy")
print(len(inputs_1), len(inputs_0))

# Create output arrays of equal length
output_0 = np.empty((len(inputs_1),), dtype=int)
output_0.fill(0)
output_1 = np.ones((len(inputs_1),), dtype=int)
output_1.fill(1)

# Trim Class 0 input data to length of Class 1 datset
inputs_0 = inputs_0[:len(inputs_1)]

# Join Class 0/CLass 1 datasets and shuffle data
inputs = np.append(inputs_0, inputs_1, axis=0)
outputs = np.append(output_0, output_1, axis=0)
inputs, outputs = shuffle(inputs, outputs)

# Normalizing data
inputs = inputs.astype('float32')
inputs = inputs / 255.0
validation_marker = 2 * int(0.15 * len(inputs))
test_marker = int(0.15 * len(inputs))

# Split into train, test and validation groups
test_inputs, test_outputs = inputs[:test_marker], outputs[:test_marker]
validation_inputs, validation_outputs = inputs[test_marker:validation_marker], outputs[test_marker:validation_marker]
train_inputs, train_outputs = inputs[validation_marker:], outputs[validation_marker:]

# Data augmentation layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(50, 50, 3)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1)
    ]
)

# Create the Machine Learning Model
model = tf.keras.Sequential([
    # Data augmentation layers
    data_augmentation,
    # Base convolutional model
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                           kernel_constraint=tf.keras.constraints.max_norm(3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                           kernel_constraint=tf.keras.constraints.max_norm(3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                           kernel_constraint=tf.keras.constraints.max_norm(3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Reduce down to one layer with a sigmoid activation as used in binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model for binary classification
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.Precision()])


# Run the model
epochs = 10
history = model.fit(
    train_inputs, train_outputs,
    validation_data=(validation_inputs, validation_outputs),
    epochs=epochs,
    batch_size=32
)

# Retrieve the relevant statistics from the learning programme
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Graphing the loss and accuracy with matplotlib for a visual analysis of training
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Run evaluation of model on test data
print("Evaluate")
result = model.evaluate(test_inputs, test_outputs)


# model.save(f"{file_dir}/cancer_model_2.h5")

# Model with binary crossentropy loss and sigmoid activation in last layer
# Info: Batch size = 32, epochs = 10, optimizer= adam, activation= relu
# Training: Loss=0.5858, accuracy=0.8151
# Validation: Loss=0.5848, accuracy=0.8091
# Note: Weird sharp fall in accuracy in epoch 5, validation nearly always better than training

# Test results: Loss=0.5918, accuracy=0.7896
