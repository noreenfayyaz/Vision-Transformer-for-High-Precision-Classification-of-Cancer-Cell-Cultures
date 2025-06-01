#importing the Libraries
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve


# Specify the path to your image dataset
#dataset_path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//MDA231_B"
dataset_path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//PC3B_Final//PC3B"
print("\n\n dataset_path : ", dataset_path)

# List to store image data and corresponding labels
images = []
labels = []

# Image height and width
height = 224
width = 224
num_classes = 9

# List of class names
#class_names =["MDA23110k01D1","MDA23110k01D2","MDA23110k01D3","MDA23110k02D1","MDA23110k02D2","MDA23110k02D3","MDA23120k01D1","MDA23120k01D2","MDA23120k01D3","MDA23120k02D1","MDA23120k02D2","MDA23120k02D3","MDA23140k01D1","MDA23140k01D2", "MDA23140k01D3","MDA23140k02D1","MDA23140k02D2", "MDA23140k02D3"]
class_names = ["PC3_10kD1","PC3_10kD2","PC3_10kD3","PC3_20kD1","PC3_20kD2","PC3_20kD3","PC3_40kD1","PC3_40kD2", "PC3_40kD3"]


# Loop through each class folder
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    print("\n\nclass_path:",class_path)
    # Loop through each image in the class folder
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)

        # Read and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (height, width))  # Resize the image to your desired dimensions
        img = img / 255.0  # Normalize pixel values to the range [0, 1]

        # Append the preprocessed image and label to the lists
        images.append(img)
        labels.append(class_names.index(class_name))  # Assuming class names are ordered numerically
print("\n\n Dataset Imported !!")

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# printing number of images and labels
print("\n\n number of images: ", len(images))
print("\n\n number of Labels: ", len(labels))

# Split the entire dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n\n X_train_shape:", X_train.shape, "\n\n y_train_shape:", y_train.shape, "\n\n X_test Shape:", X_test.shape, "\n\n y_test shape:", y_test.shape, "\n\n")

# Define the CNN model
# Function to create the CNN model
def create_model(optimizer='sgd', dropout_rate=0.5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n\n model created and intialised\n\n")

# Now, we train the model with fixed hyperparameters (without GridSearch)
optimizer = 'sgd'  # Example optimizer
dropout_rate = 0.5  # Example dropout rate

# Create the model
model = create_model(optimizer=optimizer, dropout_rate=dropout_rate)

# Train the model
history = model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_test, y_test))

# Model evaluation on the training set
train_predictions_raw = model.predict(X_train)
train_predictions = np.argmax(train_predictions_raw, axis=1)

# Accuracy on training set
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Accuracy on training set: {train_accuracy}")

# Confusion matrix for training set
conf_matrix_train = confusion_matrix(y_train, train_predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Training Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("Confusion Matrix - Training Set.png")
plt.show()

# Model evaluation on the testing set
test_predictions_raw = model.predict(X_test)
test_predictions = np.argmax(test_predictions_raw, axis=1)

# Accuracy on testing set
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy on testing set: {test_accuracy}")

# Confusion matrix for testing set
conf_matrix_test = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Testing Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("Confusion Matrix - Testing Set.png")
plt.show()

# Classification report for training and testing sets
print("\nClassification Report on Training Set:")
print(classification_report(y_train, train_predictions, target_names=class_names))

print("\nClassification Report on Testing Set:")
print(classification_report(y_test, test_predictions, target_names=class_names))

# Plot ROC curve
print("\n\n ROC's for each class")

y_test_binary = label_binarize(y_test, classes=range(num_classes))
y_test_pred_proba = model.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_micro, tpr_micro, _ = roc_curve(y_test_binary.ravel(), y_test_pred_proba.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

plt.figure(figsize=(10, 8))
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})', color='deeppink', linestyle=':')
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ROC_CNN-Testing Set.png')

# Plot accuracy and loss graph
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('accuracy_graph.png')

# Plot loss graph
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('loss_graph.png')
