import os
import numpy as np
import tensorflow as tf

# Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    print("Model Done")
    return model

# Dimensions of images and number of classes
img_width, img_height = 224, 224  # Adjust as needed
num_classes = 10  # Adjust based on your dataset

# Paths to the training and test data
train_data_path = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\dataset\new images latest\train"
test_data_path = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\dataset\new images latest\test"

# Create a mapping from class folder names to integer labels
class_folders = os.listdir(train_data_path)
class_mapping = {class_folder: idx for idx, class_folder in enumerate(class_folders)}

# Load and preprocess training images
print("hr")
train_images = []
train_labels = []
for class_folder, class_idx in class_mapping.items():
    class_folder_path = os.path.join(train_data_path, class_folder)
    for image_file in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_file)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        train_images.append(img_array)
        train_labels.append(class_idx)

print("Hi")
X_train = np.array(train_images)
y_train = tf.keras.utils.to_categorical(train_labels, num_classes)

# Load and preprocess test images
test_images = [os.path.join(test_data_path, f) for f in os.listdir(test_data_path) if f.endswith('.jpg')]
X_test = np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=(img_width, img_height))) / 255.0 for img in test_images])
print("test")
# Define callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Compile the model
model = create_cnn_model((img_width, img_height, 3), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=16, callbacks=[checkpoint, early_stopping])

# Load the best model
best_model = tf.keras.models.load_model('best_model.h5')

# Predict on test data
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Predicted classes: {predicted_classes}")
