import React from 'react';

const ReportsPage = ({ reportIndex, setReportIndex }) => {
    // List of charts to display with descriptions for a better user experience
    const chartData = [
        {
            filename: 'overallqual_vs_saleprice.png',
            title: 'Overall Quality vs. Sale Price',
            description: 'This scatter plot shows the relationship between the overall quality of a house and its sale price. Higher quality generally correlates with a higher price.'
        },
        {
            filename: 'grlivarea_vs_saleprice.png',
            title: 'Gross Living Area vs. Sale Price',
            description: 'This scatter plot visualizes how the size of a house (in square feet) affects its sale price. As expected, a larger living area tends to result in a higher price.'
        },
        {
            filename: 'feature_correlation_heatmap.png',
            title: 'Feature Correlation Heatmap',
            description: 'This heatmap shows how strongly each feature in the model is related to others. Brighter colors indicate a stronger correlation, which helps us understand our data better.'
        },
        {
            filename: 'saleprice_distribution.png',
            title: 'Sale Price Distribution',
            description: 'This histogram shows the distribution of sale prices in the training data. The data is slightly skewed, which is a common pattern in real-world housing prices.'
        },
        {
            filename: 'garage_cars_pie_chart.png',
            title: 'Garage Car Capacity Distribution',
            description: 'This pie chart shows the percentage of houses with 0, 1, 2, or more cars in their garage. It helps visualize the most common garage sizes in the dataset.'
        }
    ];

    const currentChart = chartData[reportIndex];
    const totalCharts = chartData.length;

    const handleNext = () => {
        setReportIndex((prevIndex) => (prevIndex + 1) % totalCharts);
    };

    const handlePrevious = () => {
        setReportIndex((prevIndex) => (prevIndex - 1 + totalCharts) % totalCharts);
    };

    return (
        <div className="reports-container">
            <h1>Model Visualizations</h1>
            <p className="page-indicator">{`Chart ${reportIndex + 1} of ${totalCharts}`}</p>
            <div className="chart-item">
                <h2>{currentChart.title}</h2>
                <img src={`/reports/${currentChart.filename}`} alt={currentChart.title} />
                <p className="chart-description">{currentChart.description}</p>
            </div>
            <div className="report-navigation">
                <button onClick={handlePrevious} disabled={reportIndex === 0} className="nav-button">Previous</button>
                <button onClick={handleNext} disabled={reportIndex === totalCharts - 1} className="nav-button">Next</button>
            </div>
        </div>
    );
};

export default ReportsPage;