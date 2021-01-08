
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Data from: Henry Z. Lo. and Cohen, Joseph Paul “Academic Torrents: Scalable Data Distribution.” Neural Information Processing Systems Challenges in Machine Learning (CiML) Workshop, 2016, http://arxiv.org/abs/1603.04395.
# Cohen, Joseph Paul, and Henry Z. Lo. “Academic Torrents: A Community-Maintained Distributed Repository.” Annual Conference of the Extreme Science and Engineering Discovery Environment, 2014, http://doi.org/10.1145/2616498.2616528.

# Load in pre-made numpy arrays of images
file_dir = r"C:\Users\Finn\Manual_Python\Data_Science\Cancer_ML"

inputs = np.load(f"{file_dir}/all_inputs_shuffled.npy")
outputs = np.load(f"{file_dir}/all_outputs_shuffled.npy")

# Normalizing data
inputs = inputs.astype('float32')
inputs = inputs/255.0

# Split into train, test and validation groups
validation_marker = 2*int(0.15 * len(inputs))
test_marker = int(0.15 * len(inputs))
test_inputs, test_outputs = inputs[:test_marker], outputs[:test_marker]
validation_inputs, validation_outputs = inputs[test_marker:validation_marker], outputs[test_marker:validation_marker]
train_inputs, train_outputs = inputs[validation_marker:], outputs[validation_marker:]

# Data augmentation layers
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(50,50,3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1)
  ]
)


model = tf.keras.Sequential([
    # Data augmentation layers
    data_augmentation,
    # Base convolutional model
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=10
history = model.fit(
  train_inputs,train_outputs,
  validation_data=(validation_inputs, validation_outputs),
  epochs=epochs,
    batch_size=32
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print("Evaluate")
result = model.evaluate(test_inputs, test_outputs)

epochs_range = range(epochs)

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

model.save(f"{file_dir}/cancer_model.h5")

# Results with no data augmentation or dropout layers:
# Info: Batch size = no batch size specified, epochs = 10, optimizer= adam, activation= relu
# Training - Loss=0.2688, accuracy=0.8871
# Validation - Loss=0.3135, accuracy=0.8732
# Note: Validation accuracy plateaued while training data improved

# Results with only data augmentation but no dropout layers:
# Info: Batch size = no batch size specified, epochs = 10, optimizer= adam, activation= relu
# Training - Loss=0.3081, accuracy=0.8702
# Validation - Loss=0.2970, accuracy=0.8765
# Note: Validation accuracy followed training data much more closely with data augmentation

# Results with data augmentation and dropout layer before flattening the data:
# Info: Batch size = no batch size specified, epochs = 10, optimizer= adam, activation= relu
# Training - Loss=0.3091, accuracy=0.8700
# Validation - Loss=0.3154, accuracy=0.8647
# Note: Dropout led to decrease in accuracy, can see the accuracy moving above and below training data
# Dropout causes greater variance in accuracy which is to be expected

# Results with data augmentation and dropout layer with batch-size of 32:
# Info: Batch size = 32, epochs = 10, optimizer= adam, activation= relu
# Training - Loss=0.3100, accuracy=0.8687
# Validation - Loss=0.3337, accuracy=0.8565
# Note: Validation accuracy took a sharp fall on the last epoch, validation accuracy before that was 0.8763

# Same as above with random contrast added:
# Info: Batch size = 32, epochs = 10, optimizer= adam, activation= relu
# Training - Loss=0.3123, accuracy=0.8684
# Validation - Loss=0.2966, accuracy=0.8747
# Note: Not much to note here except a weird spike in loss in the 5th epoch

# Same as above but run over 30 epochs:
# Info: Batch size = 32, epochs = 30, optimizer= adam, activation= relu
# Training - Loss=0.2928, accuracy=0.8772
# Validation - Loss=0.2921, accuracy=0.8782
# Note: Maintained a validation accuracy of over 0.88 for 4 epochs but dropped on the last epoch

# Back to 10 epochs with a kernel constraint of MaxNorm(3):
# Info: Batch size = 32, epochs = 30, optimizer= adam, activation= relu
# Training - Loss=0.3090, accuracy=0.8702
# Validation - Loss=0.2923, accuracy=0.8802
# Note:

# Test data - loss: 0.2912 - accuracy: 0.8779