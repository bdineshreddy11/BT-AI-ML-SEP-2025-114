import React from 'react';

const HomePage = ({ setCurrentPage }) => {
    return (
        <div className="home-container">
            <header className="hero-section">
                <h1 className="hero-headline">Find Your Dream Home's True Value</h1>
                <p className="hero-subtitle">
                    Our machine learning tool gives you instant, accurate house price predictions, helping you make smarter real estate decisions.
                </p>
                <button
                    className="call-to-action-button"
                    onClick={() => setCurrentPage('prediction')}
                >
                    Get Your Instant Prediction
                </button>
            </header>

            <section className="value-section">
                <h2 className="value-headline">Why Trust Our Predictor?</h2>
                <div className="value-grid">
                    <div className="value-card">
                        <h3 className="value-title">Data-Driven Accuracy</h3>
                        <p className="value-description">Our model is trained on a comprehensive dataset to provide highly reliable price estimates.</p>
                    </div>
                    <div className="value-card">
                        <h3 className="value-title">Fast & Easy to Use</h3>
                        <p className="value-description">Simply enter a few key details and get your prediction in seconds.</p>
                    </div>
                    <div className="value-card">
                        <h3 className="value-title">Informed Decisions</h3>
                        <p className="value-description">Whether you’re buying or selling, our tool provides the insight you need to negotiate with confidence.</p>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default HomePage;