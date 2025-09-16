import React from 'react';

const Header = ({ currentPage, setCurrentPage }) => {
    return (
        <header className="site-header">
            <div className="logo-container">
                <h1 className="site-logo">BT-AI-ML Project</h1>
            </div>
            <nav className="site-nav">
                <button
                    onClick={() => setCurrentPage('home')}
                    className={currentPage === 'home' ? 'nav-button active' : 'nav-button'}
                >
                    Home
                </button>
                <button
                    onClick={() => setCurrentPage('about')}
                    className={currentPage === 'about' ? 'nav-button active' : 'nav-button'}
                >
                    About
                </button>
                <button
                    onClick={() => setCurrentPage('prediction')}
                    className={currentPage === 'prediction' ? 'nav-button active' : 'nav-button'}
                >
                    Prediction
                </button>
            </nav>
        </header>
    );
};

export default Header;