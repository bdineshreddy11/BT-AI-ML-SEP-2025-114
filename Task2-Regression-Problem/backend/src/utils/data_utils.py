import pandas as pd
import joblib
import os

def load_model():
    """
    Loads the trained model from the 'models' directory.
    This function is used by the prediction service to make predictions.
    """
    model_path = os.path.join('src', 'models', 'house_price_model.pkl')
    try:
        model = joblib.load(model_path)
        print("ML model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run 'train_and_save_model.py' first.")
        return None

def preprocess_input(data: dict):
    """
    Example function to preprocess incoming JSON data from the frontend.
    This must match the preprocessing done during training.
    """
    # Create a DataFrame from the input data.
    processed_df = pd.DataFrame([data])
    return processed_df