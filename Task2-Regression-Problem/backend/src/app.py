from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing
# The import path below is correct, assuming you run the app from the 'backend' folder
from src.services.prediction_service import get_house_prediction
import os

# Create the Flask application instance.
app = Flask(__name__)
# Enable CORS for all routes, allowing the frontend to communicate with the backend.
CORS(app)

@app.route('/')
def health_check():
    """A simple health check endpoint to verify the backend is running."""
    return jsonify({"status": "Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive house features and return a price prediction.
    It expects a JSON payload from the frontend.
    """
    # Check if the request body is valid JSON.
    if request.is_json:
        data = request.get_json()
        print(f"Received data for prediction: {data}")
        # Call the prediction service to get the result.
        prediction_result = get_house_prediction(data)
        return jsonify(prediction_result)
    else:
        # Return an error if the request is not in JSON format.
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    # This block is for running the app directly, for development purposes.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
