from src.utils.data_utils import load_model, preprocess_input
import numpy as np

# Load the model once when the service starts to improve performance.
global_model = load_model()

def get_house_prediction(features: dict):
    """
    Takes a dictionary of house features and returns a predicted price.
    """
    # Check if the model was loaded successfully.
    if global_model is None:
        return {"error": "ML model not loaded."}, 500

    try:
        # Preprocess the incoming features exactly as the model expects.
        processed_features = preprocess_input(features)

        # Make the prediction using the loaded model.
        prediction = global_model.predict(processed_features)

        # Return the prediction. The result is a numpy array, so we convert it to a float.
        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Failed to make prediction: {e}"}, 500