import numpy as np
import pandas as pd
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
    shuffle=False)

x_test = image_dataset_from_directory(
    '/app/rundir/cpsc542_assignment1/jellyfish_dataset/Train_Test_Valid/test',
    image_size=(128, 128),
    label_mode='int',
    batch_size=32,
    shuffle=False)

# ----- Random Forest (Non Deep Learning Model) -----
# extract images and labels from datasets
train_images = []
train_labels = []

for images, labels in x_train:
    train_images.extend(images.numpy())
    train_labels.extend(labels.numpy())

test_images = []
test_labels = []

for images, labels in x_test:
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())

# convert the data into a pandas dataframe
train_df = pd.DataFrame({
    'image': train_images,
    'label': train_labels
})

test_df = pd.DataFrame({
    'image': test_images,
    'label': test_labels
})

# convert lists of images to numpy arrays
X_train = np.array(train_df['image'].tolist())
y_train = train_df['label']

X_test = np.array(test_df['image'].tolist())
y_test = test_df['label']

# flatten images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# create and train decision tree model
# dt = DecisionTreeClassifier(random_state = 42)
# dt.fit(X_train_flat, y_train)

# create and train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train)

# make predictions 
y_pred = rf.predict(X_test_flat)
accuracy_train = accuracy_score(y_train, y_train)
accuracy_test = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print('Random Forest Train Accuracy: ', accuracy_train)
print('Random Forest Test Accuracy: ', accuracy_test)
print('Random Forest Model Precision: ', precision)
print('Random Forest Model Recall: ', recall)

# confusion matrices
classes = ['Barrel', 'Blue', 'Compass', 'Lion\'s Mane', 'Mauve Stinger', 'Moon']

# train confusion matrix
cm = confusion_matrix(y_train, rf.predict(X_train_flat))
weights = class_weight.compute_sample_weight('balanced', y_train)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('RF Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('visualizations/rf/cm_train_rf.png')

# test confusion matrix
cm = confusion_matrix(y_test, y_pred)
weights = class_weight.compute_sample_weight('balanced', y_test)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('RF Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('visualizations/rf/cm_test_rf.png')

# visualize predictions
label_mapping = {
    0: 'Barrel',
    1: 'Blue',
    2: 'Compass',
    3: 'Lion\'s Mane',
    4: 'Mauve Stinger',
    5: 'Moon'
}

y_pred_names = [label_mapping[label] for label in y_pred]
y_test_names = [label_mapping[label] for label in y_test]

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15),
    gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

for i, ax in enumerate(axes.flat):
    # plot image
    ax.imshow(X_test[i].reshape(128, 128, 3).astype('uint8')) 
    ax.axis('off')

    # set title to predicted class and true class
    pred_class = y_pred_names[i]
    true_class = y_test_names[i]
    ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}",
        color=("green" if pred_class == true_class else "red"))

plt.savefig('visualizations/rf/vis_rf.png')