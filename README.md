# AgroMitra

## Crop Disease Detection Web Application

AgroMitra is a web application for detecting crop diseases using a Convolutional Neural Network (CNN) model. This application allows users to upload images of crops and receive a diagnosis of the disease affecting the crop along with suggested solutions or preventive measures.

### Features

- Upload crop images to detect diseases.
- Provides the name of the disease and possible solutions or preventive measures.
- Built using TensorFlow and Keras for training the CNN model.
- Web interface developed using HTML, CSS (LESS), JavaScript, and MySQL.
- Hosted using ngrok for easy access.

### Project Structure

- `app.py`: Main Flask application file to run the web server.
- `CropDiseaseDetector.py`: Contains the logic for loading the model and predicting crop diseases.
- `GetSolution.py`: Provides solutions and preventive measures for detected diseases.
- `static/`: Contains static files like CSS, JavaScript, and images.
- `templates/`: Contains HTML templates for rendering web pages.
- `models/`: Directory where trained models are saved.

### Requirements

- Python 3.x
- TensorFlow
- Keras
- Flask
- Ngrok
- MySQL

### Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AgroMitra.git
   cd AgroMitra
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MySQL Database**
   - Create a MySQL database and user.
   - Update the database configuration in `app.py`.

4. **Train the Model**
   - Update the dataset paths in the code.
   - Run the training script to train the model.
   - The trained model will be saved in the `models/` directory.
   ```bash
   python train_model.py
   ```

5. **Start the Flask Application**
   ```bash
   python app.py
   ```

6. **Expose Local Server Using Ngrok**
   ```bash
   ngrok http 5000
   ```

### Usage

1. Open the ngrok URL in a web browser.
2. Upload a crop image.
3. View the predicted disease name and suggested solutions.

### Code Explanation

#### `train_model.py`

This script loads and preprocesses the dataset, builds a CNN model, trains the model, and saves it in both `.h5` and `.tflite` formats.

#### `app.py`

Main Flask application file that handles routes for the web interface, file uploads, and interactions with the CNN model and database.

#### `CropDiseaseDetector.py`

Contains the logic for loading the trained CNN model and predicting the disease from uploaded crop images.

#### `GetSolution.py`

Provides solutions or preventive measures for the detected crop diseases based on predefined mappings.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgements

- TensorFlow and Keras for providing the tools to build and train the CNN model.
- Flask for the web framework.
- Ngrok for easy web hosting and tunneling.
- MySQL for database management.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

---

Feel free to reach out if you have any questions or need further assistance. Thank you for using AgroMitra!
