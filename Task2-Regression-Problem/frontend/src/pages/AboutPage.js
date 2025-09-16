import React from 'react';

const AboutPage = () => {
    return (
        <div style={{ padding: '20px', textAlign: 'center' }}>
            <h1>About This Project</h1>
            <p>
                This project was created to demonstrate a full-stack machine learning application. The frontend is built with React, and the backend uses Python's Flask framework to serve a trained house price prediction model.
            </p>
            <p>
                The model was trained on the House Prices - Advanced Regression Techniques dataset from Kaggle.
            </p>
        </div>
    );
};

export default AboutPage;
