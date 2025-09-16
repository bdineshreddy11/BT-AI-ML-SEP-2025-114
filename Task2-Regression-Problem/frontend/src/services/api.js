import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export const predictHousePrice = async (features) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict`, features);
        return response.data;
    } catch (error) {
        console.error("Error making prediction API call:", error);
        throw error;
    }
};