import React from 'react';

const PredictionDisplay = ({ prediction, error }) => {
    if (error) {
        return (
            <div className="error-message fade-in">
                Error: {error}
            </div>
        );
    }

    if (prediction === null) {
        return null;
    }

    const formattedPrediction = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(prediction);

    return (
        <div className="prediction-result fade-in">
            <h3>Predicted House Price:</h3>
            <p>{formattedPrediction}</p>
        </div>
    );
};

export default PredictionDisplay;