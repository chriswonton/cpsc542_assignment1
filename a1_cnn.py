import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# load train and test datasets
x_train = image_dataset_from_directory(
    '/app/rundir/cpsc542_assignment1/jellyfish_dataset/Train_Test_Valid/Train',
    image_size=(128, 128),
    label_mode='int',
    batch_size=32,
    shuffle=True)

x_test = image_dataset_from_directory(
    '/app/rundir/cpsc542_assignment1/jellyfish_dataset/Train_Test_Valid/test',
    image_size=(128, 128),
    label_mode='int',
    batch_size=32,
    shuffle=False)

# data Augmentation
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

X_train, y_train = np.concatenate([x for x, y in x_train], axis=0), np.concatenate([y for x, y in x_train], axis=0)
X_test, y_test = np.concatenate([x for x, y in x_test], axis=0), np.concatenate([y for x, y in x_test], axis=0)

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ----- CNN (Deep Learning Model) -----

# use MobileNetV2 as base model for transfer learning
base_model = MobileNetV2(include_top=False, input_tensor=Input(shape=(128, 128, 3)))

# freeze the layers of the base model
base_model.trainable = False

# add custom layers on top of MobileNetV2
cnn = kb.Sequential([
    base_model,
    kb.layers.GlobalAveragePooling2D(),
    kb.layers.Dropout(0.5),
    kb.layers.Dense(256, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(6, activation='softmax')
])

# compile the model with an initial learning rate
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
cnn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train the model with early stopping
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = cnn.fit(x_train, batch_size=32, epochs=100, validation_data=x_test)

# plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('visualizations/cnn/accuracy_cnn')

# cnn.save('cnn_trained.h5')
# cnn = load_model('cnn_trained.h5')

# confusion matrices
classes = ['Barrel', 'Blue', 'Compass', 'Lion\'s Mane', 'Mauve Stinger', 'Moon']

# training confusion matrix
y_pred_probs_train = cnn.predict(X_train)
y_pred_labels_train = np.argmax(y_pred_probs_train, axis=1)
y_true = np.argmax(y_train, axis=1)

cm = confusion_matrix(y_true, y_pred_labels_train)
weights = class_weight.compute_sample_weight('balanced', y_true)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10,8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('CNN Training Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('visualizations/cnn/cm_train_cnn.png')

# test confusion matrix
y_pred_probs_test = cnn.predict(X_test)
y_pred_labels_test = np.argmax(y_pred_probs_test, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_labels_test)
weights = class_weight.compute_sample_weight('balanced', y_true)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10,8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('CNN Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('visualizations/cnn/cm_test_cnn.png')

label_mapping = {
    0: 'Barrel',
    1: 'Blue',
    2: 'Compass',
    3: 'Lion\'s Mane',
    4: 'Mauve Stinger',
    5: 'Moon'
}

# get some test images
images, labels = next(iter(x_test))
images = images * np.max(images) * 255

# make predictions
predictions = cnn.predict(images)

# visualize predictions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15),
    gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

for i, ax in enumerate(axes.flat):
    # plot image
    ax.imshow(images[i].numpy().astype('uint8'))
    ax.axis('off')

    # set title to predicted class and true class using label mapping
    pred_class = np.argmax(predictions[i])
    true_class = np.argmax(labels[i])
    pred_class_name = label_mapping[pred_class]
    true_class_name = label_mapping[true_class]
    ax.set_title(f"Pred: {pred_class_name}\nTrue: {true_class_name}",
        color=("green" if pred_class == true_class else "red"))
    
plt.savefig('visualizations/cnn/vis_cnn.png')