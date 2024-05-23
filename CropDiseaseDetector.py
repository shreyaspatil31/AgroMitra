import numpy as np
from PIL import Image
import tensorflow as tf

class CropDiseaseDetector:
    def __init__(self):
        self.models = {
            "RICE": "./static/model/rice.tflite",
            "GRAPE": "./static/model/grape.tflite",
            "SUGARCANE": "./static/model/sugarcane.tflite"
        }
        self.disease = {
            "RICE": {
                0: "Bacterial Leaf Blight",
                1: "Brown Spot",
                2: "Rice Leaf Smut"
            },
            "GRAPE": {
                0: "Grape Black Rot",
                1: "Grape Esca (Black Measles)",
                2: "Grape Healthy"
            },
            "SUGARCANE":{
                0: "Sugarcane Healthy",
                1: "Sugarcane Mosaic",
                2: "REDDOT",
                3: "Sugarcane Rust",
                4: "Sugarcane Yellow"
            }
        }
        self.interpreters = {crop: tf.lite.Interpreter(model_path=model_path) for crop, model_path in self.models.items()}
        for interpreter in self.interpreters.values():
            interpreter.allocate_tensors()

    def preprocess_image(self, image_path, target_size):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = img_array.astype(np.float32)
        return img_array

    def predict_disease(self, image_array, interpreter, crop):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        output_tensor = interpreter.get_tensor(output_details[0]['index'])

        output_copy = np.array(output_tensor, copy=True)

        predicted_class_index = np.argmax(output_copy)
        disease_name = self.disease[crop].get(predicted_class_index, "Unknown Disease")

        del image_array
        del output_tensor
        del output_copy

        return predicted_class_index, disease_name

    def detect_disease(self, image_path, crop):
        processed_image = self.preprocess_image(image_path, (255, 255))
        interpreter_to_use = self.interpreters[crop]
        predicted_class, disease_name = self.predict_disease(np.expand_dims(processed_image, axis=0), interpreter_to_use, crop)
        predicted_class = int(predicted_class)
        return predicted_class, disease_name
