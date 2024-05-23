import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

IMAGE_SIZE = (255, 255)
BATCH_SIZE = 8
CHANNELS = 3
NUM_DISPLAY_IMAGES = 9

# PHASE 1: LOAD & PREPROCESS DATA

# Function to load and preprocess data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(selected_dataset, image_size, batch_size, test_split=0.2):
    # Create a generator for training and testing data with data augmentation
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=test_split  # Set test split
    )
    
    # Create a dataset
    dataset = data_generator.flow_from_directory(
        directory=selected_dataset,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'  # Specify this as training data
    )

    return dataset

# Function to extract crop and disease names
def extract_crop_and_disease_names(class_name):
    parts = class_name.split("/")
    crop_name = dataset_directory
    disease_name = parts[0]

    return crop_name, disease_name


# Function to display images
def display_images(dataset, num_images=NUM_DISPLAY_IMAGES):
    plt.figure(figsize=(10, 10))
    class_names = list(dataset.class_indices.keys())
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        batch = next(dataset)
        images, labels = batch
        plt.imshow(images[0], aspect='auto', interpolation='nearest')
        class_index = np.argmax(labels[0])
        class_name = class_names[class_index]
        crop_name, disease_name = extract_crop_and_disease_names(class_name)
        plt.title(f'Crop: {crop_name}\nDisease: {disease_name}')
        plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()

# Define dataset paths
dataset_path = {
    "GRAPE": "D:/PROJECT/MAJOR PROJECT/NEW_DATASET/Grape",
    "RICE": "D:/PROJECT/MAJOR PROJECT/NEW_DATASET/Rice",
    "SUGARCANE": "D:/PROJECT/MAJOR PROJECT/NEW_DATASET/Sugarcane"
}

# Display available options to the user
print("Available datasets:")
for key in dataset_path:
    print(key)

# Ask the user for input
dataset_directory = input("Enter the name of the crop you want to select: ")

# PHASE 1 FINISH

# PHASE 2: BUILD AND TRAIN CNN MODEL

# Function to build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Function to train the CNN model and return intermediate layer outputs
def train_cnn_model(model, train_dataset, num_epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=num_epochs
    )

    return model, history

# Plot training history
def plot_metrics(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history['accuracy'], label='Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history['precision'], label='Precision')
    plt.plot(epochs, history.history['val_precision'], label='Val Precision')
    plt.title('Precision over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.history['recall'], label='Recall')
    plt.plot(epochs, history.history['val_recall'], label='Val Recall')
    plt.title('Recall over Epochs')
    plt.legend()

    plt.show()


# Your main function
def main():
    if dataset_directory in dataset_path:
        selected_dataset = dataset_path[dataset_directory]
        if not os.path.exists(selected_dataset):
            print(f"Error: Dataset path '{selected_dataset}' does not exist.")
            return

        # Load training data
        train_dataset = load_and_preprocess_data(selected_dataset, IMAGE_SIZE, BATCH_SIZE)

        # Check if dataset loading was successful
        if not train_dataset:
            print("Failed to load dataset.")
            return
        
        # Create the CNN model
        num_classes = len(train_dataset.class_indices)
        input_shape = IMAGE_SIZE + (CHANNELS,)
        model = build_cnn_model(input_shape, num_classes)

        # Train the model and get intermediate layer outputs
        trained_model, history = train_cnn_model(model, train_dataset, num_epochs=10)

        # Evaluate the model on the training dataset
        test_loss, test_accuracy = trained_model.evaluate(train_dataset)
        print(f"Test Accuracy: {test_accuracy}")

        # Save the trained model for the specific crop
        trained_model.save(f"D:/PROJECT/MAJOR PROJECT/{dataset_directory}_model.h5")
        
        # Convert and save the trained model to .tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(f"D:/PROJECT/MAJOR PROJECT/{dataset_directory}_model.tflite", 'wb') as file:
            file.write(tflite_model)

        # Get the final training metrics
        final_training_loss = history.history['loss'][-1]
        final_training_accuracy = history.history['accuracy'][-1]

        print("Final Training Loss:", final_training_loss)
        print("Final Training Accuracy:", final_training_accuracy)
        
        # Show class names and indices
        print("\nClass Names and Indices:")
        for class_name, class_index in train_dataset.class_indices.items():
            print(f"{class_name}: {class_index}")
        
        
        # Evaluate model to get precision, recall, F1-score
        Y_pred = trained_model.predict(train_dataset, train_dataset.samples // BATCH_SIZE + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(train_dataset.classes, y_pred))
        print('Classification Report')
        target_names = list(train_dataset.class_indices.keys())
        print(classification_report(train_dataset.classes, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
