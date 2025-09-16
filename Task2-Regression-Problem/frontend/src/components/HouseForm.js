import React, { useState } from 'react';

const HouseForm = ({ onSubmit, isLoading }) => {
    const [formData, setFormData] = useState({
        GrLivArea: '',
        GarageCars: '',
        OverallQual: '',
        TotalBsmtSF: '',
        YearBuilt: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        const numericData = Object.fromEntries(
            Object.entries(formData).map(([key, value]) => [key, parseFloat(value)])
        );
        onSubmit(numericData);
    };

    return (
        <form onSubmit={handleSubmit} className="fade-in">
            <div className="form-group">
                <label htmlFor="GrLivArea">Gross Living Area (sqft):</label>
                <input
                    type="number"
                    id="GrLivArea"
                    name="GrLivArea"
                    value={formData.GrLivArea}
                    onChange={handleChange}
                    required
                    min="0"
                />
            </div>
            <div className="form-group">
                <label htmlFor="GarageCars">Garage Car Capacity:</label>
                <input
                    type="number"
                    id="GarageCars"
                    name="GarageCars"
                    value={formData.GarageCars}
                    onChange={handleChange}
                    required
                    min="0"
                    max="4"
                />
            </div>
            <div className="form-group">
                <label htmlFor="OverallQual">Overall Quality (1-10):</label>
                <input
                    type="number"
                    id="OverallQual"
                    name="OverallQual"
                    value={formData.OverallQual}
                    onChange={handleChange}
                    required
                    min="1"
                    max="10"
                />
            </div>
            <div className="form-group">
                <label htmlFor="TotalBsmtSF">Total Basement SF (sqft):</label>
                <input
                    type="number"
                    id="TotalBsmtSF"
                    name="TotalBsmtSF"
                    value={formData.TotalBsmtSF}
                    onChange={handleChange}
                    required
                    min="0"
                />
            </div>
            <div className="form-group">
                <label htmlFor="YearBuilt">Year Built:</label>
                <input
                    type="number"
                    id="YearBuilt"
                    name="YearBuilt"
                    value={formData.YearBuilt}
                    onChange={handleChange}
                    required
                    min="1800"
                    max={new Date().getFullYear()}
                />
            </div>
            <button type="submit" className="submit-button" disabled={isLoading}>
                {isLoading ? 'Predicting...' : 'Get Prediction'}
            </button>
        </form>
    );
};

export default HouseForm;