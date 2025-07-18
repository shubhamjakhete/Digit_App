# Handwritten Digit Recognition Web App

A modern web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application features a clean, dark-themed UI for uploading and processing handwritten digit images.

## Features

- **Deep Learning Model**: Custom CNN architecture trained on MNIST dataset
- **Modern Web Interface**: Dark-themed, responsive UI with drag-and-drop functionality
- **Real-time Processing**: Instant digit recognition with confidence scores
- **Monitoring System**: Logging system for predictions and error tracking
- **RESTful API**: Flask backend serving predictions via JSON responses

## Project Structure

```
Digit_App/
├── training/                  # Model training code and data
│   ├── train_digit_model.py  # CNN model definition and training script
│   └── data/                 # MNIST dataset (downloaded automatically)
├── model/                    # Saved model weights
│   └── improved_digit_model.pth
├── api/                      # Flask backend
│   └── app.py               # API endpoints and model serving
├── web/                      # Frontend
│   └── templates/
│       └── index.html       # Main web interface
├── monitor/                  # Monitoring and logging
│   └── logger.py            # Prediction logging system
└── requirements.txt         # Python dependencies
```

## Technology Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, torchvision
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Image Processing**: Pillow
- **Monitoring**: Python logging

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Digit_App.git
   cd Digit_App
   ```

2. **Create Virtual Environment**
   ```bash
   cd api
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install flask torch torchvision pillow numpy
   ```

4. **Train the Model**
   ```bash
   cd ../training
   python train_digit_model.py
   ```
   This will:
   - Download the MNIST dataset
   - Train the CNN model
   - Save the model weights to `model/improved_digit_model.pth`

5. **Start the Flask Server**
   ```bash
   cd ../api
   python app.py
   ```

6. **Access the Web Interface**
   - Open your browser
   - Navigate to `http://localhost:5000`
   - Start recognizing digits!

## Model Architecture

The CNN architecture consists of:
- 3 Convolutional layers with batch normalization
- Max pooling and dropout for regularization
- 2 Fully connected layers
- Output layer for 10 digits (0-9)

Training parameters:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 20 (with early stopping)

## API Endpoints

1. **GET /** 
   - Returns the main web interface

2. **POST /predict**
   - Accepts: Image file via FormData
   - Returns: JSON with predicted digit
   ```json
   {
     "digit": 5
   }
   ```

## Monitoring and Logging

The application logs:
- Predictions with confidence scores
- Error messages
- Processing times

Logs are stored in:
- `monitor/predictions.log`
- Console output (during development)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/
- Flask Documentation: https://flask.palletsprojects.com/
