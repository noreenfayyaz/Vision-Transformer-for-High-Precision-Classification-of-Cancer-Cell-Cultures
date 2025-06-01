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
import tensorflow_addons as tfa
from tensorflow.keras import layers
import random, cv2, os, glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import datetime as dt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from itertools import product

print("showing the path:")
#path = "//mmfs1//projects//changhui.yan//noreen.f.khan//anomoly_detection//Dataset//Dataset_small//*"
path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//MDA231_Final//MDA231_B//*"
print("\n\n dataset_path : ",path)

data_paths = os.path.join(path,'*g')
imagePaths = glob.glob(data_paths)

dim = (224, 224)
labels=[]
images=[]
image_size = 224
num_classes = 9
input_shape = (224, 224, 3)
image_size = 224  # We'll resize input images to this size
patch_size = 12  # Size of the patches to be extracted from the input images
projection_dim = 64
num_epochs = 200

i=0
for imgpath in imagePaths:
    #print(imgpath)
    frame=cv2.imread(imgpath)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    #label= imgpath.split(os.path.sep)[-2].split("_")
    label = imgpath.split(os.path.sep)[-2]
    images.append(frame)
    labels.append(label)

images = np.array(images)

labels
len(images)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print("\n\n # of labels: ", len(labels))
print("\n # of images : ", len(images))

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.2)

print("\n\n train_set: ", len(x_train))
print("\n\n test_set: ", len(x_test))
print("\n\n train_labels: ", len( y_train))
print("\n\n test_labels: ", len(y_test))

print(f"\n x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


print("\n\n Data Augmentation process: ")
data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02)
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
    
    
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


print("\n\n plotting figure")
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
# plt.savefig(image.astype("uint8"))
plt.savefig("original_img_vit.png")
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(224, 224)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    # plt.savefig(patch_img.numpy().astype("uint8"))
    plt.savefig("patchimage_vit_kfold.png")
    plt.axis("off")
    
print ("\n\n patch Encoder")    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
  
#Define hyperparameter grid
#param_grid = {
#   'learning_rate': [0.001, 0.0001],
#   'batch_size': [32, 64],
#   'transformer_layers': [4, 8],
#   'num_heads': [5, 7]
#}

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.001],
    'batch_size': [64],
    'transformer_layers': [8],
    'num_heads': [7]
}

# Function to create and compile the model
def create_vit_classifier(params):
   
    num_patches = (image_size // patch_size) ** 2
    num_heads = params['num_heads']
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = params['transformer_layers']
    mlp_head_units = [512, 512]  # Size of the dense layers of the final classifier
    learning_rate = params['learning_rate']
    weight_decay = 0.0001
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model
print("\n\n model created and intialised\n\n")



# Function to run 5-fold cross-validation for given hyperparameters
def run_cv_experiment(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(images):
        x_train, x_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = create_vit_classifier(params)
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=params['batch_size'],
            epochs=num_epochs,
            validation_data=(x_val, y_val),
            verbose=0
        )
       

        val_accuracy = history.history['val_accuracy'][-1]
        accuracies.append(val_accuracy)

    return np.mean(accuracies)

# Grid search loop
results = []

for params in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    print("Training with parameters:", param_dict)

    mean_val_accuracy = run_cv_experiment(param_dict)

    # Store results
    results.append({
        'params': param_dict,
        'mean_val_accuracy': mean_val_accuracy
    })

    #print("Finished training with parameters:", param_dict)

# Print results
for result in results:
    print("Parameters:", result['params'])
    print("Mean Validation Accuracy:", result['mean_val_accuracy'])
    
# Find the best parameters
best_result = max(results, key=lambda x: x['mean_val_accuracy'])
# Print the best parameters and mean validation accuracy
print("\n\n Best Parameters:", best_result['params'])
print("\n\n Mean Validation Accuracy:", best_result['mean_val_accuracy'])


import matplotlib.pyplot as plt
print("\n\n retrain model with best parameters:")
# Function to retrain model with best parameters
def retrain_best_model(best_params):
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)  # Split data into train and validation sets

    # Create and compile the model with best parameters
    best_model = create_vit_classifier(best_params)

    # Train the model
    history = best_model.fit(
        x=x_train,
        y=y_train,
        batch_size=best_params['batch_size'],
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )
    return best_model, history

# Retrain model with best parameters
best_params = best_result['params']
best_model, retrain_history = retrain_best_model(best_params)

print("\n\n Accuracy and Loss Graph")
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.plot(retrain_history.history['accuracy'], 'o-')
ax.plot(retrain_history.history['val_accuracy'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.savefig("Accuracy_plot_VIT_Kfold.png",  dpi=300, bbox_inches='tight')

f, ax = plt.subplots()
ax.plot(retrain_history.history['loss'], 'o-')
ax.plot(retrain_history.history['val_loss'], 'x-')

# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train loss', "Val loss"], loc = 1)
ax.set_title('Training Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
#plt.show()
plt.savefig("Loss1_plot_VIT_Kfold.png",  dpi=300, bbox_inches='tight')


# Evaluate the retrained model on test
evaluation = best_model.evaluate(x_test, y_test)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# Evaluate the retrained model on trainset
evaluation_train = best_model.evaluate(x_train, y_train)
print("Train Loss:", evaluation_train[0])
print("Train Accuracy:", evaluation_train[1])

print("\n\n best_ model: ", best_model)

print("\n\n Plotting Confusion Matrix2 ")
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions for the training set
train_predictions_raw = best_model.predict(x_train)
train_predictions = np.argmax(train_predictions_raw, axis=1)

# Get predictions for the testing set
test_predictions_raw = best_model.predict(x_test)
test_predictions = np.argmax(test_predictions_raw, axis=1)

# Define target names based on the label encoder
target_names = label_encoder.classes_

# Generate and print classification report for the training set
print("\n\nClassification Report on trainset")
print(classification_report(y_train, train_predictions, target_names=target_names))


# Generate and print classification report for the testing set
print("\n\nClassification Report on testset")
print(classification_report(y_test, test_predictions, target_names=target_names))



# Generating ROC
print("\n\n ROC's for each class")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Ensure labels are one-hot encoded
y_test_one_hot = label_binarize(y_test, classes=np.arange(num_classes))

# Get predicted probabilities
y_pred_proba = best_model.predict(x_test)

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc="lower right")
plt.savefig("ROC_curves_VIT_Kfold.png",  dpi=300, bbox_inches='tight')  # Save the ROC curves plot
plt.show()


print("\n\n ROC_plot ends")



print("\n\n Generate confusion matrix on Training set\n")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Get predictions for the training set
train_predictions_raw = best_model.predict(x_train)
train_predictions = np.argmax(train_predictions_raw, axis=1)

# Get predictions for the testing set
test_predictions_raw = best_model.predict(x_test)
test_predictions = np.argmax(test_predictions_raw, axis=1)

# Calculate confusion matrices
train_cm = confusion_matrix(y_train, train_predictions)
test_cm = confusion_matrix(y_test, test_predictions)

# Plot confusion matrix for the training set
plt.figure(figsize=(20, 15))  # Increase figure size
disp = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Training Set', fontsize=10)
plt.xticks(fontsize=10, rotation=45)  # Increase x-tick label size and rotate for better visibility
plt.yticks(fontsize=10)               # Increase y-tick label size
plt.tight_layout()                    # Adjust layout to make sure labels are not cut off
plt.savefig("Confusion_Matrix_Training_VIT_Kfold.png",  dpi=300, bbox_inches='tight')  # Save the plot
plt.show()

# Plot confusion matrix for the testing set
plt.figure(figsize=(20, 15))  # Increase figure size
disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Testing Set', fontsize=10)
plt.xticks(fontsize=10, rotation=45)  # Increase x-tick label size and rotate
plt.yticks(fontsize=10)               # Increase y-tick label size
plt.tight_layout()                    # Adjust layout to make sure labels are fully visible
plt.savefig("Confusion_Matrix_Testing_VIT_Kfold.png",  dpi=300, bbox_inches='tight')  # Save the plot
plt.show()

