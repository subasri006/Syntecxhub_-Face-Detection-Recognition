/**
 * Main Frontend Application JavaScript
 * Controls tab navigation, WebRTC camera streaming, API integration, and UI state.
 */

// Global App State
let activeTab = 'dashboard';
let apiBaseUrl = CONFIG.getApiBaseUrl();

// Live Recognition State
let liveStream = null;
let liveInterval = null;
let liveFacingMode = 'user'; // 'user' (front) or 'environment' (rear)
let liveThreshold = CONFIG.DEFAULT_THRESHOLD;

// Registration State
let regStream = null;
let regSamples = [];

// Image Detection State
let loadedImageB64 = null;
let processedImageB64 = null;
let detectThreshold = CONFIG.DEFAULT_THRESHOLD;

// Selected person for deletion
let targetDeletePerson = null;

// Initialization on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSliders();
    initImageDetection();
    initRegistration();
    checkBackendHealth();

    console.log(`[Frontend] Initialized API Base URL: ${apiBaseUrl}`);
});

// -----------------------------------------------------------------------------
// NAVIGATION & TABS
// -----------------------------------------------------------------------------
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    if (activeTab === 'live' && tabId !== 'live') {
        stopLiveCamera();
    }
    if (activeTab === 'register' && tabId !== 'register') {
        stopRegCamera();
    }

    activeTab = tabId;

    document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.getAttribute('data-tab') === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    document.querySelectorAll('.tab-page').forEach(page => {
        if (page.id === `tab-${tabId}`) {
            page.classList.add('active');
        } else {
            page.classList.remove('active');
        }
    });

    if (tabId === 'dashboard') {
        checkBackendHealth();
    } else if (tabId === 'people') {
        loadPeopleList();
    }
}

// -----------------------------------------------------------------------------
// BACKEND HEALTH & METRICS
// -----------------------------------------------------------------------------
async function checkBackendHealth() {
    const badgeApi = document.getElementById('badge-api');
    const badgeDb = document.getElementById('badge-db');
    const dashStatus = document.getElementById('dash-backend-status');
    const dashCount = document.getElementById('dash-count-people');

    try {
        const response = await fetch(`${apiBaseUrl}/health`);
        if (!response.ok) throw new Error('Health check failed');
        const data = await response.json();

        badgeApi.className = 'badge badge-success';
        badgeApi.textContent = '● Backend: Connected';

        badgeDb.textContent = `● Registered: ${data.database_people_count}`;
        if (dashStatus) dashStatus.textContent = 'Operational';
        if (dashCount) dashCount.textContent = data.database_people_count;

        loadDashboardPeople();
    } catch (err) {
        console.warn('[Backend Health Error]', err);
        badgeApi.className = 'badge badge-warning';
        badgeApi.textContent = '● Backend: Offline / Error';
        if (dashStatus) dashStatus.textContent = 'Offline';
    }
}

// -----------------------------------------------------------------------------
// SLIDERS & CONTROLS
// -----------------------------------------------------------------------------
function initSliders() {
    const sliderLive = document.getElementById('live-threshold');
    const valLive = document.getElementById('live-thresh-val');
    if (sliderLive) {
        sliderLive.addEventListener('input', (e) => {
            liveThreshold = parseFloat(e.target.value);
            valLive.textContent = liveThreshold.toFixed(3);
        });
    }

    const sliderDetect = document.getElementById('detect-threshold');
    const valDetect = document.getElementById('detect-thresh-val');
    if (sliderDetect) {
        sliderDetect.addEventListener('input', (e) => {
            detectThreshold = parseFloat(e.target.value);
            valDetect.textContent = detectThreshold.toFixed(3);
            if (loadedImageB64) {
                processSelectedImage();
            }
        });
    }

    // Live Camera Buttons
    document.getElementById('btn-start-cam').addEventListener('click', startLiveCamera);
    document.getElementById('btn-stop-cam').addEventListener('click', stopLiveCamera);
    document.getElementById('btn-switch-cam').addEventListener('click', switchLiveCamera);
}

// -----------------------------------------------------------------------------
// LIVE RECOGNITION (WebRTC + BASE64 STREAMING)
// -----------------------------------------------------------------------------
async function startLiveCamera() {
    if (liveStream) return;

    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('video-placeholder');
    const btnStart = document.getElementById('btn-start-cam');
    const btnStop = document.getElementById('btn-stop-cam');
    const badgeStatus = document.getElementById('live-status-badge');

    try {
        const constraints = {
            video: {
                facingMode: liveFacingMode,
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        liveStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = liveStream;
        placeholder.style.display = 'none';

        btnStart.disabled = true;
        btnStop.disabled = false;
        badgeStatus.className = 'badge badge-success';
        badgeStatus.textContent = 'Status: Streaming Live';

        // Start processing loop
        liveInterval = setInterval(processLiveFrame, CONFIG.FRAME_CAPTURE_INTERVAL_MS);

    } catch (err) {
        console.error('[Webcam Error]', err);
        alert(`Unable to access camera: ${err.message}\nPlease ensure camera permissions are granted.`);
    }
}

async function processLiveFrame() {
    if (!liveStream) return;

    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const imgAnnotated = document.getElementById('webcam-annotated');
    const faceCountBadge = document.getElementById('live-face-count');
    const namesText = document.getElementById('live-names');

    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const b64Frame = canvas.toDataURL('image/jpeg', 0.8);

    try {
        const response = await fetch(`${apiBaseUrl}/api/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: b64Frame,
                threshold: liveThreshold
            })
        });

        if (!response.ok) return;

        const data = await response.json();

        // Update annotated image overlay
        imgAnnotated.src = data.annotated_image;

        // Update stats
        faceCountBadge.textContent = `Faces Detected: ${data.face_count}`;

        if (data.faces && data.faces.length > 0) {
            const identities = data.faces.map(f => `${f.name} (${f.similarity_score.toFixed(2)})`).join(', ');
            namesText.textContent = `Identities: ${identities}`;
        } else {
            namesText.textContent = 'Identities: None';
        }

    } catch (err) {
        console.warn('[Live Processing Error]', err);
    }
}

function stopLiveCamera() {
    if (liveInterval) {
        clearInterval(liveInterval);
        liveInterval = null;
    }

    if (liveStream) {
        liveStream.getTracks().forEach(track => track.stop());
        liveStream = null;
    }

    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('video-placeholder');
    const imgAnnotated = document.getElementById('webcam-annotated');
    const btnStart = document.getElementById('btn-start-cam');
    const btnStop = document.getElementById('btn-stop-cam');
    const badgeStatus = document.getElementById('live-status-badge');

    video.srcObject = null;
    imgAnnotated.src = '';
    placeholder.style.display = 'flex';

    btnStart.disabled = false;
    btnStop.disabled = true;
    badgeStatus.className = 'badge badge-secondary';
    badgeStatus.textContent = 'Status: Offline';
}

function switchLiveCamera() {
    liveFacingMode = (liveFacingMode === 'user') ? 'environment' : 'user';
    if (liveStream) {
        stopLiveCamera();
        startLiveCamera();
    }
}

// -----------------------------------------------------------------------------
// IMAGE DETECTION
// -----------------------------------------------------------------------------
function initImageDetection() {
    const fileInput = document.getElementById('img-file-input');
    const btnSave = document.getElementById('btn-save-img');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
            loadedImageB64 = evt.target.result;
            processSelectedImage();
        };
        reader.readAsDataURL(file);
    });

    btnSave.addEventListener('click', downloadProcessedImage);
}

async function processSelectedImage() {
    if (!loadedImageB64) return;

    const imgDisplay = document.getElementById('img-display');
    const placeholder = document.getElementById('img-placeholder');
    const summaryCount = document.getElementById('detect-summary-count');
    const resultsList = document.getElementById('detect-results-list');
    const btnSave = document.getElementById('btn-save-img');

    placeholder.style.display = 'none';
    imgDisplay.style.display = 'block';

    resultsList.textContent = 'Processing image with YuNet & SFace...';

    try {
        const response = await fetch(`${apiBaseUrl}/api/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: loadedImageB64,
                threshold: detectThreshold
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Image processing failed');
        }

        const data = await response.json();

        processedImageB64 = data.annotated_image;
        imgDisplay.src = processedImageB64;
        summaryCount.textContent = `Total Faces Detected: ${data.face_count}`;

        if (data.faces && data.faces.length > 0) {
            const details = data.faces.map((f, i) => 
                `Face #${i+1}:\n  Identity: ${f.name}\n  Cosine Similarity: ${f.similarity_score}\n  Detection Confidence: ${f.detection_score}\n`
            ).join('\n');
            resultsList.textContent = details;
        } else {
            resultsList.textContent = 'No faces were detected in the provided image.';
        }

        btnSave.disabled = false;

    } catch (err) {
        console.error('[Image Processing Error]', err);
        resultsList.textContent = `Error: ${err.message}`;
    }
}

function downloadProcessedImage() {
    if (!processedImageB64) return;
    const a = document.createElement('a');
    a.href = processedImageB64;
    a.download = 'face_detection_result.jpg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// -----------------------------------------------------------------------------
// REGISTER PERSON
// -----------------------------------------------------------------------------
function initRegistration() {
    const btnStart = document.getElementById('btn-reg-start-cam');
    const btnSnap = document.getElementById('btn-reg-snap');
    const btnReset = document.getElementById('btn-reg-reset');
    const btnCancel = document.getElementById('btn-reg-cancel');

    btnStart.addEventListener('click', startRegCamera);
    btnSnap.addEventListener('click', captureRegSample);
    btnReset.addEventListener('click', resetRegSamples);
    btnCancel.addEventListener('click', stopRegCamera);
}

async function startRegCamera() {
    const name = document.getElementById('reg-name-input').value.trim();
    const consent = document.getElementById('reg-consent-chk').checked;
    const statusMsg = document.getElementById('reg-status-msg');

    if (!name) {
        alert('Please enter the full name of the person.');
        return;
    }

    if (!consent) {
        alert('User consent is required prior to capturing face data.');
        return;
    }

    const video = document.getElementById('reg-video');
    const placeholder = document.getElementById('reg-placeholder');

    try {
        regStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });

        video.srcObject = regStream;
        placeholder.style.display = 'none';

        regSamples = [];
        updateRegUI();

        document.getElementById('btn-reg-start-cam').disabled = true;
        document.getElementById('btn-reg-snap').disabled = false;
        document.getElementById('btn-reg-reset').disabled = false;
        document.getElementById('btn-reg-cancel').disabled = false;

        statusMsg.textContent = 'Position your face clearly. Ensure ONLY 1 face is visible.';
        statusMsg.style.color = 'var(--primary)';

    } catch (err) {
        console.error('[Registration Camera Error]', err);
        alert(`Camera access failed: ${err.message}`);
    }
}

function captureRegSample() {
    if (!regStream) return;

    const video = document.getElementById('reg-video');
    const canvas = document.getElementById('reg-canvas');
    const statusMsg = document.getElementById('reg-status-msg');

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const b64Sample = canvas.toDataURL('image/jpeg', 0.85);

    regSamples.push(b64Sample);
    updateRegUI();

    statusMsg.textContent = `Captured sample ${regSamples.length} of 5.`;
    statusMsg.style.color = 'var(--success)';

    if (regSamples.length >= 5) {
        submitRegistration();
    }
}

function resetRegSamples() {
    regSamples = [];
    updateRegUI();
    document.getElementById('reg-status-msg').textContent = 'Samples reset. Capture 5 new samples.';
}

function updateRegUI() {
    document.getElementById('reg-sample-num').textContent = regSamples.length;
}

async function submitRegistration() {
    const name = document.getElementById('reg-name-input').value.trim();
    const consent = document.getElementById('reg-consent-chk').checked;
    const statusMsg = document.getElementById('reg-status-msg');

    statusMsg.textContent = 'Submitting samples & building face embedding template...';
    statusMsg.style.color = 'var(--purple)';

    try {
        const response = await fetch(`${apiBaseUrl}/api/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                samples: regSamples,
                consent: consent
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Registration failed');
        }

        alert(`Registration Successful!\nRegistered '${name}' with 5 samples.`);
        stopRegCamera();
        switchTab('people');

    } catch (err) {
        console.error('[Registration Submit Error]', err);
        alert(`Registration Error: ${err.message}`);
        statusMsg.textContent = `Error: ${err.message}`;
        statusMsg.style.color = 'var(--danger)';
    }
}

function stopRegCamera() {
    if (regStream) {
        regStream.getTracks().forEach(track => track.stop());
        regStream = null;
    }

    const video = document.getElementById('reg-video');
    const placeholder = document.getElementById('reg-placeholder');

    video.srcObject = null;
    placeholder.style.display = 'flex';

    document.getElementById('btn-reg-start-cam').disabled = false;
    document.getElementById('btn-reg-snap').disabled = true;
    document.getElementById('btn-reg-reset').disabled = true;
    document.getElementById('btn-reg-cancel').disabled = true;

    regSamples = [];
    updateRegUI();
    document.getElementById('reg-status-msg').textContent = 'Enter full name, check consent, then start camera.';
    document.getElementById('reg-status-msg').style.color = 'var(--text-muted)';
}

// -----------------------------------------------------------------------------
// REGISTERED PEOPLE MANAGER
// -----------------------------------------------------------------------------
async function loadPeopleList() {
    const tbody = document.getElementById('people-table-body');
    tbody.innerHTML = '<tr><td colspan="3" class="text-center">Loading people from database...</td></tr>';

    try {
        const response = await fetch(`${apiBaseUrl}/api/people`);
        if (!response.ok) throw new Error('Failed to fetch people');

        const people = await response.json();
        tbody.innerHTML = '';

        if (people.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">No registered people in database.</td></tr>';
            return;
        }

        people.forEach(person => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${escapeHtml(person.name)}</strong></td>
                <td>${person.sample_count} samples</td>
                <td>
                    <button class="btn btn-danger btn-sm" onclick="promptDeletePerson('${escapeHtml(person.name)}')">
                        🗑️ Delete
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('[People List Error]', err);
        tbody.innerHTML = `<tr><td colspan="3" class="text-center text-danger">Error loading list: ${err.message}</td></tr>`;
    }
}

async function loadDashboardPeople() {
    const tbody = document.getElementById('dash-table-body');
    if (!tbody) return;

    try {
        const response = await fetch(`${apiBaseUrl}/api/people`);
        if (!response.ok) return;

        const people = await response.json();
        tbody.innerHTML = '';

        if (people.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">No registered users yet.</td></tr>';
            return;
        }

        people.forEach(person => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${escapeHtml(person.name)}</strong></td>
                <td>${person.sample_count} samples</td>
                <td><span class="badge badge-success">Ready</span></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.warn('[Dashboard People Error]', err);
    }
}

function promptDeletePerson(name) {
    targetDeletePerson = name;
    document.getElementById('admin-modal').style.display = 'flex';
}

document.getElementById('btn-close-modal').addEventListener('click', () => {
    document.getElementById('admin-modal').style.display = 'none';
    targetDeletePerson = null;
});

document.getElementById('btn-confirm-delete').addEventListener('click', async () => {
    if (!targetDeletePerson) return;

    const apiKey = document.getElementById('admin-key-input').value.trim();

    try {
        const response = await fetch(`${apiBaseUrl}/api/people/${encodeURIComponent(targetDeletePerson)}`, {
            method: 'DELETE',
            headers: {
                'X-API-Key': apiKey
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to delete person');
        }

        alert(`Successfully deleted '${targetDeletePerson}'.`);
        document.getElementById('admin-modal').style.display = 'none';
        document.getElementById('admin-key-input').value = '';
        targetDeletePerson = null;

        loadPeopleList();
        checkBackendHealth();

    } catch (err) {
        alert(`Delete Error: ${err.message}`);
    }
});

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
