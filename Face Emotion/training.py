import numpy as np
import pandas as pd
import os
import gc
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# === Label mapping ===
id_to_emotion_mapper = {0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Natural',6:'Sad',7:'Sleepy',8:'Surprised'}
mapping_to_new = {0:0,1:-1,2:-1,3:4,4:1,5:2,6:3,7:2,8:4}
new_emotion_mapper = {0:'Angry',1:'Happy',2:'Neutral',3:'Sad',4:'Surprised'}

# === Root folder ===
root = '/kaggle/input/8-facial-expressions-for-yolo/9 Facial Expressions you need'

# === Dataset generator ===
def dataset_generator(dataset='training', aug=True):
    dataset = dataset.lower()
    if dataset == 'training':
        folder = 'train'
    elif dataset == 'validation':
        folder = 'valid'
    elif dataset == 'testing':
        folder = 'test'
    else:
        print('\nError: wrong dataset/folder name provided\n')
        return -1
    
    images, labels = [], []
    available = [6000, 6000, 6000, 6000, 6000]
    
    dir_folder = os.path.join(root, folder)
    dir_folder_images = os.path.join(dir_folder, 'images')
    dir_folder_labels = os.path.join(dir_folder, 'labels')
    
    label_list = os.listdir(dir_folder_labels)
    for label_filename in label_list:
        label_path = os.path.join(dir_folder_labels, label_filename)
        with open(label_path) as f:
            line = f.readline().strip().split()
            if not line: 
                continue
            label_idx = int(line[0])
            new_idx = mapping_to_new[label_idx]
            if new_idx >= 0 and available[new_idx] >= 0:
                oneHot_label = [0, 0, 0, 0, 0]
                oneHot_label[new_idx] = 1
                img_path_jpg = os.path.join(dir_folder_images, label_filename.replace('.txt', '.jpg'))
                img_path_png = os.path.join(dir_folder_images, label_filename.replace('.txt', '.png'))
                img_path = img_path_jpg if os.path.exists(img_path_jpg) else (img_path_png if os.path.exists(img_path_png) else None)
                if img_path:
                    available[new_idx] -= 1
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(oneHot_label)
    
    images = np.array(images)
    labels = np.array(labels)
    print(f'Shape of images: {images.shape}, labels: {labels.shape}')
    
    temp_map = {0:0, 1:0, 2:0, 3:0, 4:0}
    for i in labels:
        temp_map[np.argmax(i)] += 1
    print("\nLabel distribution:\n", temp_map)
    
    if aug and len(images) > 0:
        generator = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=10,
            horizontal_flip=True,
            zoom_range=0.1,
            preprocessing_function=preprocess_input
        )
        dataset = generator.flow(images, labels)
        del images, labels
        gc.collect()
        return dataset
    else:
        return images, labels


# === Data Preparation ===
train_dataset = dataset_generator('training')
val_dataset = dataset_generator('validation')
test_images, test_labels = dataset_generator('testing', aug=False)


# === Model ===
base_model = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')]
)

# === Callbacks ===
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1)
]

# === Training ===
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,  # no early stopping, run all epochs
    callbacks=callbacks
)

# === Accuracy Graph ===
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# === Evaluate on Test Data ===
predictions = model.predict(preprocess_input(test_images))
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(test_labels, axis=1)

loss, accuracy, auc = model.evaluate(preprocess_input(test_images), test_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(new_emotion_mapper.values()))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# === Detailed Metrics ===
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(new_emotion_mapper.values())))

# === Save model ===
model.save('face_model_final.h5')
