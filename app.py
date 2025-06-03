from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model from root
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join('samples')  # Using your 'samples' folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('static/uploads', filename)  # Save to static/uploads
    file.save(filepath)


    # ----- Image Preprocessing -----
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))  
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
  


    prediction = model.predict(img)
    label = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'

    return render_template('result.html', prediction=label, image='uploads/' + filename)


if __name__ == '__main__':
    app.run(debug=True)

