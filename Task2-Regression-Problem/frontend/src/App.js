import React, { useState } from 'react';
import './index.css';
import HouseForm from './components/HouseForm';
import PredictionDisplay from './components/PredictionDisplay';
import { predictHousePrice } from './services/api';
import AboutPage from './pages/AboutPage';
import Header from './components/Header';
import Footer from './components/Footer';
import ReportsPage from './pages/ReportsPage';
import HomePage from './pages/HomePage';

function App() {
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentPage, setCurrentPage] = useState('home');
    const [showVisualizations, setShowVisualizations] = useState(false);
    const [reportIndex, setReportIndex] = useState(0);

    const handleFormSubmit = async (formData) => {
        setIsLoading(true);
        setPrediction(null);
        setError(null);
        setShowVisualizations(false);

        try {
            const result = await predictHousePrice(formData);
            if (result.predicted_price) {
                setPrediction(result.predicted_price);
                setShowVisualizations(true);
            } else if (result.error) {
                setError(result.error);
            } else {
                setError("Unknown error occurred.");
            }
        } catch (err) {
            console.error("Failed to fetch prediction:", err);
            setError("Failed to connect to the prediction service. Please try again later.");
        } finally {
            setIsLoading(false);
        }
    };

    const renderPageContent = () => {
        switch (currentPage) {
            case 'home':
                return <HomePage setCurrentPage={setCurrentPage} />;
            case 'prediction':
                return (
                    <div className="prediction-page-container">
                        <h1>House Price Predictor</h1>
                        <p>Enter the features of the house to get an estimated price.</p>
                        {isLoading && <div className="loading-spinner"></div>}
                        {!isLoading && <HouseForm onSubmit={handleFormSubmit} isLoading={isLoading} />}
                        <PredictionDisplay prediction={prediction} error={error} />
                        {showVisualizations && (
                            <div style={{marginTop: '20px'}}>
                                <button onClick={() => { setCurrentPage('reports'); setReportIndex(0); }} className="visualize-button">
                                    Show Visualizations
                                </button>
                            </div>
                        )}
                    </div>
                );
            case 'about':
                return <AboutPage />;
            case 'reports':
                return <ReportsPage reportIndex={reportIndex} setReportIndex={setReportIndex} />;
            default:
                return null;
        }
    };

    return (
        <>
            <Header currentPage={currentPage} setCurrentPage={setCurrentPage} />
            <div className="main-content">
                {renderPageContent()}
            </div>
            <Footer />
        </>
    );
}

export default App;