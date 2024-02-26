import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input

import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# load train and test datasets
x_train = image_dataset_from_directory(
    '/app/rundir/assignment1/jellyfish_dataset/Train_Test_Valid/Train',
    image_size=(128, 128),
    label_mode='int',
    batch_size=32,
    shuffle=True)

x_test = image_dataset_from_directory(
    '/app/rundir/assignment1/jellyfish_dataset/Train_Test_Valid/test',
    image_size=(128, 128),
    label_mode='int',
    batch_size=32,
    shuffle=True)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

def preprocess_data(x, y):
    x = data_augmentation(x)
    return x, tf.one_hot(y, 6)

x_train = x_train.map(preprocess_data)
x_test = x_test.map(preprocess_data)

# ----- CNN (Deep Learning Model) -----

# Use MobileNetV2 as base model
base_model = MobileNetV2(include_top=False, input_tensor=Input(shape=(128, 128, 3)))

# Freeze the layers of the base model
base_model.trainable = False

# Add custom layers on top of MobileNetV2
cnn = kb.Sequential([
    base_model,
    kb.layers.GlobalAveragePooling2D(),
    kb.layers.Dropout(0.5),
    kb.layers.Dense(256, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(6, activation='softmax')
])

# Compile the model with an initial learning rate
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
cnn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Class Weighting
# class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = cnn.fit(x_train, batch_size=32, epochs=100, validation_data=x_test)

# Evaluate the model on the test set
# test_loss, test_accuracy = cnn.evaluate(x_test)
# print("CNN Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_cnn')

# cnn.save('cnn_trained.h5')
# cnn = load_model('cnn_trained.h5')

y_pred = cnn.predict(x_test).argmax(axis=1)
y_true = np.concatenate([y for x, y in x_test], axis=0).argmax(axis=1)

# accuracy = accuracy_score(y_true, y_pred)
# print('CNN Accuracy:', accuracy)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN Confusion Matrix for Test Dataset')
plt.savefig('cm_cnn_test.png')
