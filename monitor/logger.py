import logging
import os

def setup_logger():
    """
    Sets up the logger for monitoring digit predictions
    """
    # Create the monitor directory if it doesn't exist
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__))):
        os.makedirs(os.path.dirname(os.path.abspath(__file__)))
    
    # Configure the logger
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add a stream handler to also print logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def log_prediction(image_name, prediction, confidence=None):
    """
    Logs a prediction with optional confidence score
    
    Args:
        image_name (str): Name of the input image
        prediction (int): Predicted digit
        confidence (float, optional): Confidence score of the prediction
    """
    if confidence is not None:
        logging.info(f'Prediction for {image_name}: {prediction} (confidence: {confidence:.2f})')
    else:
        logging.info(f'Prediction for {image_name}: {prediction}')

def log_error(error_message):
    """
    Logs an error message
    
    Args:
        error_message (str): Error message to log
    """
    logging.error(f'Error: {error_message}') 