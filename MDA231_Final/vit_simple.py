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
#from mtcnn.mtcnn import MTCNN
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
#from mtcnn.mtcnn import MTCNN
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

print("showing the path:")
# path = "//mmfs1//projects//changhui.yan//noreen.f.khan//anomoly_detection//Dataset//Dataset_small//*"
path = "//mmfs1//projects//changhui.yan//noreen.f.khan//cell_dataset//MDA231_Final//MDA231_B//*"
print("\n\n dataset_path : ",path)

data_paths = os.path.join(path,'*g')
imagePaths = glob.glob(data_paths)

dim = (224, 224)
labels=[]
images=[]
#detector = MTCNN()
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


num_classes = 9
input_shape = (224, 224, 3)
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 64
num_epochs = 200
image_size = 224 # We'll resize input images to this size
patch_size = 12  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 5
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [512, 512]  # Size of the dense layers of the final classifier

print("\n\n Data Augmentation process: ")
data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
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
plt.savefig("original_img.png")
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
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
    plt.savefig("patchimage.png")
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
  
  

print("\n\n Defining VIT Simple Model")
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model    
    
print("\n\n model created and intialised\n\n")

print("\n\n running the model")
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2
    )


    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

#print("\n\n vit_classifier summary: ", vit_classifier.summary())

print("\n\n Accuracy and Loss Graph")
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.plot([None] + history.history['accuracy'], 'o-')
ax.plot([None] + history.history['val_accuracy'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.savefig("Training_Accuracy1.png")

f, ax = plt.subplots()
ax.plot([None] + history.history['loss'], 'o-')
ax.plot([None] + history.history['val_loss'], 'x-')

# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train loss', "Val loss"], loc = 1)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
#plt.show()
plt.savefig("Training_Loss1.png")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
batch_size = 64
target_names = ["MDA231_10kD1","MDA231_10kD2","MDA231_10kD3","MDA231_20kD1","MDA231_20kD2","MDA231_20kD3","MDA231_40kD1","MDA231_40kD2", "MDA231_40kD3"]

print("\n\nPlotting Confusion Matrix and Classification Report on Training Dataset")

# Predict on the training dataset
Y_pred_train = vit_classifier.predict(x_train, batch_size=batch_size)  # Updated for TensorFlow 2.x
y_pred_train = np.argmax(Y_pred_train, axis=1)

# Generate Confusion Matrix and Classification Report
cm_train = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix (Train Dataset):")
print(cm_train)

print("\nClassification Report (Train Dataset):")
print(classification_report(y_train, y_pred_train, target_names=target_names))

# Normalize Confusion Matrix
cmn_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(20, 7))
sns.heatmap(cmn_train, annot=True, fmt=".2f", linewidths=1, cmap="Greens",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (Train Dataset)")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


print("\n\n Plotting Confusion Matrix-2 and Classification report2")
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


batch_size=64
from sklearn.metrics import classification_report, confusion_matrix
num_of_test_samples = 1800
# target_names = ["PC3 10k 01 D1","PC3 10k 01 D2","PC3 10k 01 D3","PC3 10k 02 D1","PC3 10k 02 D2","PC3 10k 02 D3","PC3 20k 01 D1","PC3 20k 01 D2","PC3 20k 01 D3","PC3 20k 02 D1","PC3 20k 02 D2","PC3 20k 02 D3","PC3 40k 01 D1","PC3 40k 01 D2", "PC3 40k 01 D3","PC3 40k 02 D1","PC3 40k 02 D2", "PC3 40k 02 D3"]
target_names = ["MDA231_10kD1","MDA231_10kD2","MDA231_10kD3","MDA231_20kD1","MDA231_20kD2","MDA231_20kD3","MDA231_40kD1","MDA231_40kD2", "MDA231_40kD3"]
#Confution Matrix and Classification Report
Y_pred = vit_classifier.predict_generator(x_test, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=target_names))
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(20,7))

sns.heatmap(cmn, center=0, annot=True, fmt='.2f', linewidths=1,  xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.show()

