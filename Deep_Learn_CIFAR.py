import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data and split into train/test data
(inputs_train, labels_train), (inputs_test, test_labels) = tf.keras.datasets.cifar10.load_data()

# Preprocess data
inputs_train = inputs_train.astype('float32')
inputs_test = inputs_test.astype('float32')
inputs_train = inputs_train/255.0
inputs_test = inputs_test/255.0

# Form validation dataset
Batch_size = 32
validation_size = int(0.1 * len(inputs_train))
validation_inputs, validation_labels = inputs_train[:validation_size], labels_train[:validation_size]
train_inputs, train_labels = inputs_train[validation_size:], labels_train[validation_size:]

# One hot encoding dataset
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
validation_labels = tf.keras.utils.to_categorical(validation_labels)

num_classes = train_labels.shape[1]

# Make the model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(32,32,3), padding='same', activation='relu',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(32,32,3), padding='same', activation='relu',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=(32,32,3), padding='same', activation='relu',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same',
                                 kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(512, activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=(0.01/25), nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(train_inputs, train_labels, validation_data=(validation_inputs,validation_labels), epochs=10, batch_size=32)

# Graphing the loss over epochs
loss_train = history.history['train_loss']
loss_val = history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Graphing the accuracy over epochs
acc_train = history.history['acc']
acc_val = history.history['val_acc']
epochs = range(1,11)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(inputs_test, test_labels)
model.save('cifar_model.h5')

