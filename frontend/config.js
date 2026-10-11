
/**
 * Frontend Global Configuration
 */

const CONFIG = {
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
