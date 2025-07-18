from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.train_digit_model import ImprovedCNN
from monitor.logger import setup_logger, log_prediction, log_error

app = Flask(__name__, template_folder='../web/templates')

# Set up logging
setup_logger()

# Transform for input images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model():
    model = ImprovedCNN()
    model.load_state_dict(torch.load('../model/improved_digit_model.pth',
                                   map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        if 'image' not in request.files:
            log_error('No image uploaded')
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            log_error('No image selected')
            return jsonify({'error': 'No image selected'}), 400
        
        # Open and preprocess the image
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Log the prediction
        log_prediction(file.filename, prediction, confidence)
        
        return jsonify({'digit': prediction})
    
    except Exception as e:
        error_msg = str(e)
        log_error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True) 