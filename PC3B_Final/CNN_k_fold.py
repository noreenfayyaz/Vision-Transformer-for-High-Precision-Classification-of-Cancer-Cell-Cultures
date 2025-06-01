#pip install tensorflow-addons

#importing the Libraries
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import random,cv2,os,glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime as dt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
import numpy as np
from numpy import *

# importing packages
import cv2
import os
import numpy as np

# Specify the path to your image dataset

#dataset_path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//MDA231"
dataset_path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//PC3B_Final//PC3B"
print("\n\n dataset_path : ",dataset_path)

# List to store image data and corresponding labels
images = []
labels = []

# Image height and width
height = 224
width = 224
num_classes = 9


# List of class names

class_names = ["PC3_10kD1","PC3_10kD2","PC3_10kD3","PC3_20kD1","PC3_20kD2","PC3_20kD3","PC3_40kD1","PC3_40kD2", "PC3_40kD3"]
#class_names = ["blast", "blood", "fight", "gunshots", "normal"]


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

print("\n\n number of images: ",len(images))
print("\n\n number of Labels: ",len(labels))

# !pip install scikeras[tensorflow]

from keras.wrappers.scikit_learn import KerasClassifier
#!pip install keras

#!pip install scikeras

print("\n\n start executing the model..\n\n")
#executing the model
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
import numpy as np

# Split the entire dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n\n X_train_shape:", X_train.shape,"\n\n y_train_shape:",y_train.shape,"\n\n X_test Shape:",X_test.shape,"\n\n y_test shape:",y_test.shape,"\n\n")

# Define the CNN model
# Function to create the CNN model
def create_model(optimizer='adam', dropout_rate=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n\n model created and intialised\n\n")

#print(model.summary())

# Wrap the Keras model in a scikit-learn compatible wrapper
keras_model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, verbose=1)

# Define hyperparameters to search
#param_grid = {
#    'optimizer': ['adam', 'sgd', 'rmsprop'],
#   'epochs': [10, 15, 20],  # Adjust epochs directly instead of dropout_rate
#}

param_grid = {
    'optimizer': ['adam', 'sgd'],
    'epochs': [30, 40],  # Adjust epochs directly instead of dropout_rate
}

# Create GridSearchCV
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=4, verbose=1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

best_params = grid_search.best_params_
print(f"\n\n Best hyperparameters: {grid_search.best_params_}")


# Access the detailed results of the grid search
results = grid_search.cv_results_
print(results)

# Extract and print the relevant information
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(f"\n\n Mean Score: {mean_score:.4f}, Parameters: {params}")

#Retrain Model on Best Parameters

# Create a new instance of the KerasClassifier with the best hyperparameters
best_model = KerasClassifier(create_model, batch_size=64, verbose=1, **best_params)


# Retrain the model on the entire training set
best_model.fit(X_train, y_train)

print("\n\n best_ model: ", best_model)

# Predictions on the training set
from sklearn.preprocessing import LabelBinarizer

# Convert predictions to one-hot encoded format
lb = LabelBinarizer()
# Fitting the LabelBinarizer on the training set


# Fitting the LabelBinarizer on the training set
lb.fit(y_train)

# Predictions on the training set
train_predictions_raw = best_model.predict(X_train)

# Convert predictions to one-hot encoded format
train_predictions_one_hot = lb.transform(train_predictions_raw)

# Get the predicted class labels
train_predictions = np.argmax(train_predictions_one_hot, axis=1)

# Evaluate the best model on the training set
train_accuracy = accuracy_score(y_train, train_predictions)

print(f"Accuracy on training set using the best model: {train_accuracy}")
print("\n\n Generate confusion matrix on Training set\n")
# Generate confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
conf_matrix_train = confusion_matrix(y_train, train_predictions)
print("\n\n a")
# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(20, 15))
print("\n\n b")
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
print("\n\n c")
plt.title("Confusion Matrix - Training Set")
plt.savefig("Confusion Matrix - Training Set.png", dpi=300, bbox_inches='tight')  # Save the plot
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



print ("\n\n\n Evaluate the best model on the testing set\n\n ")

# Evaluate the best model on the testing set
# Before the line causing the error
test_predictions_raw = best_model.predict(X_test)
print("Shape of test predictions:", test_predictions_raw.shape)

# Print some information about the predictions
print("Sample predictions:", test_predictions_raw[:18])

# Check the shape of y_test
print("Shape of y_test:", y_test.shape)

# Check if the dimensions are compatible
print("Dimensions of y_test after argmax:", np.argmax(test_predictions_raw).shape)

from sklearn.preprocessing import LabelBinarizer

# Convert predictions to one-hot encoded format
lb = LabelBinarizer()
test_predictions_one_hot = lb.fit_transform(test_predictions_raw)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get the predicted class labels
test_predictions = np.argmax(test_predictions_one_hot, axis=1)

# Flatten y_test if needed
y_test_flat = np.ravel(y_test)

# Evaluate the best model on the testing set
test_accuracy = accuracy_score(y_test_flat, test_predictions)

#print(f"Best model accuracy on training set: {best_accuracy}")
print(f"Accuracy on testing set using the best model: {test_accuracy}")

print("\n\n Confusion Matrix - Testing Set:")
# Generate confusion matrix for testing set


conf_matrix_test = confusion_matrix(y_test, test_predictions)

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Testing Set")
plt.savefig("Confusion Matrix - Testing Set.png",  dpi=300, bbox_inches='tight')  # Save the plot
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



print("\n\n Classification Reports on training and testing dataset: \n")
# classification report
from sklearn.metrics import classification_report

# Assuming you already have the 'best_model', 'X_train', 'X_test', 'y_train', and 'y_test' loaded

# Evaluate the best model on the training set
train_predictions = np.argmax(train_predictions_one_hot, axis=1)

# Generate a classification report for the training set
print("Classification Report on Training Set:")
print(classification_report(y_train, train_predictions, target_names=class_names))

# Evaluate the best model on the testing set
test_predictions = np.argmax(test_predictions_one_hot, axis=1)

# Generate a classification report for the testing set
print("\nClassification Report on Testing Set:")
print(classification_report(y_test, test_predictions, target_names=class_names))

# Generating ROC
print("\n\n ROC's for each class")
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Convert labels to one-hot encoding
y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# Get predicted probabilities for each class
y_test_pred_proba = best_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr_micro, tpr_micro, _ = roc_curve(y_test_binary.ravel(), y_test_pred_proba.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr_macro = all_fpr
tpr_macro = mean_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)

# Plot ROC curves
plt.figure(figsize=(20, 15))
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})', color='deeppink', linestyle=':')
plt.plot(fpr_macro, tpr_macro, label=f'Macro-average ROC curve (AUC = {roc_auc_macro:.2f})', color='navy', linestyle=':')

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('ROC_CNN-Testing Set.png', dpi=300, bbox_inches='tight')
print("\n\n ROC2_plot ends")

print("\n\n Accuaracy and Loss Graph")
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier

# Modified training process to store the training history
best_model = KerasClassifier(create_model, batch_size=64, verbose=1, **best_params)

# Retrain the model on the entire training set and store the training history
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Access the history from the Keras model
history_dict = history.model_.history.history

# Plot accuracy graph
plt.figure(figsize=(20, 15))
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('accuracy_graph.png',  dpi=300, bbox_inches='tight')  # Save the accuracy graph

# Plot loss graph
plt.figure(figsize=(20, 15))
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('loss_graph.png',  dpi=300, bbox_inches='tight')  # Save the loss graph
