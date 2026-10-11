/**
 * Frontend Global Configuration
 * Configures API base URL depending on local development vs production deployment.
 */

const CONFIG = {
    // Replace this URL with your deployed Render backend URL when deploying to Vercel
    // e.g., 'https://syntecxhub-face-recognition-backend.onrender.com'
    ```javascript
RENDER_BACKEND_URL: 'https://syntecxhub-face-detection-recognition.onrender.com',

    DEFAULT_THRESHOLD: 0.363,
    FRAME_CAPTURE_INTERVAL_MS: 350,

    getApiBaseUrl: function() {
        const hostname = window.location.hostname;
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        return this.RENDER_BACKEND_URL;
    }
};
