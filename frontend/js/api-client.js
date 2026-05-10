/**
 * API Client for AgriSurgeon
 * Handles all communication with the backend API
 */

const API_BASE_URL = 'https://akash918266-agrisurgeon-api.hf.space';

class APIClient {
    constructor(baseURL = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`);
            if (!response.ok) throw new Error('Health check failed');
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            return null;
        }
    }

    /**
     * Predict disease from image and environmental data
     */
    async predictDisease(formData) {
        try {
            const response = await fetch(`${this.baseURL}/api/predict`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    /**
     * Batch predict multiple images
     */
    async batchPredict(formData) {
        try {
            const response = await fetch(`${this.baseURL}/api/batch-predict`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Batch prediction failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Batch prediction error:', error);
            throw error;
        }
    }

    /**
     * Get list of all diseases
     */
    async getDiseases() {
        try {
            const response = await fetch(`${this.baseURL}/api/diseases`);
            if (!response.ok) throw new Error('Failed to fetch diseases');
            return await response.json();
        } catch (error) {
            console.error('Get diseases error:', error);
            return null;
        }
    }

    /**
     * Get model information
     */
    async getModels() {
        try {
            const response = await fetch(`${this.baseURL}/api/models`);
            if (!response.ok) throw new Error('Failed to fetch models');
            return await response.json();
        } catch (error) {
            console.error('Get models error:', error);
            return null;
        }
    }

    /**
     * Get API statistics
     */
    async getStatistics() {
        try {
            const response = await fetch(`${this.baseURL}/api/statistics`);
            if (!response.ok) throw new Error('Failed to fetch statistics');
            return await response.json();
        } catch (error) {
            console.error('Get statistics error:', error);
            return null;
        }
    }
}

// Create global instance
const apiClient = new APIClient();
