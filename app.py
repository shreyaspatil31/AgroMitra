import json
from flask import Flask, render_template, request, jsonify
from CropDiseaseDetector import CropDiseaseDetector
from GetSolution import get_solution

app = Flask(__name__)

@app.route("/")
def renderHomePage():
    return render_template("homePage.html")

@app.route("/detect")
def renderDetectPage():
    return render_template("detectPage.html")

@app.route('/upload', methods=['POST'])
def upload():
    crop_type = request.form['cropType']
    image_file = request.files['image']

    image_file.save("./static/image/crop.jpg")
    
    detector = CropDiseaseDetector()
    _, disease_name = detector.detect_disease("./static/image/crop.jpg", crop_type)
    solution = get_solution(crop_type, disease_name)
    
    return jsonify({"name": disease_name, "solution": solution})

if __name__ == '__main__':
    app.run(debug=True)
