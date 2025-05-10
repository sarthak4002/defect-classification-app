from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(os.path.join('model', 'defect_classification_model_rf.joblib'))

def extract_hog_features(image_path):
    # Read the image in grayscale
    image = imread(image_path, as_gray=True)
    
    # Resize the image to the same size used during training (e.g., 512x1408)
    image_resized = cv2.resize(image, (512, 1408))  # Change dimensions to match training size
    
    # Extract HOG features
    fd, hog_image = hog(image_resized, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image file found.'
    
    files = request.files.getlist('image')  # Get the list of uploaded files
    results = []  # Store results for each image

    for file in files:
        if file.filename == '':
            continue
        
        # Save the image to a temporary location
        image_path = f'temp_{file.filename}'
        file.save(image_path)

        # Preprocess the image and make predictions
        features = extract_hog_features(image_path).reshape(1, -1)
        
        # Debug information
        print("Shape of extracted features:", features.shape)  # Print shape of features

        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)  # Get prediction probabilities
        
        # Print prediction probabilities for debugging
        print("Prediction probabilities:", prediction_proba)  # Print probabilities

        result = 'Defective' if prediction[0] == 1 else 'Non-Defective'
        probability = prediction_proba[0][1] * 100 if prediction[0] == 1 else prediction_proba[0][0] * 100  # Probability as percentage
        
        # Store the result for this image
        results.append({
            'filename': file.filename,
            'result': result,
            'probability': round(probability, 2)  # Round probability to 2 decimal places
        })

        # Optionally, remove the temporary image after prediction
        os.remove(image_path)

    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
