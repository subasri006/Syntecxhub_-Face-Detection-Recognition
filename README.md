# Face Detection & Recognition System (Desktop + Mobile Web Application)

A dual-architecture **Face Detection and Recognition Application** that runs both as a **Windows Desktop App** (Tkinter + OpenCV) and a **Mobile-Friendly Web Application** (HTML/CSS/JS Frontend on Vercel + FastAPI Backend on Render).

Developed for **Syntecxhub Internship – Week 3**.

---

## 📌 Project Overview

This project provides real-time face detection, alignment, embedding extraction, and cosine similarity matching against registered identities using **OpenCV YuNet** (face detection) and **OpenCV SFace** (face recognition).

### Dual Architecture:
1. **Windows Desktop App**: Python 3.11, Tkinter, OpenCV (`main.py`).
2. **Mobile Web Application**:
   - **Frontend**: Responsive HTML5, CSS3, Vanilla JS (`frontend/`) deployed on **Vercel**.
   - **Backend**: Python FastAPI REST API (`backend/`) deployed on **Render**.

---

## 📂 Project Structure

```
Face_Detection_Recognition/
├── main.py                 # Desktop GUI Application (Tkinter)
├── register_face.py        # Desktop CLI: Register person via webcam
├── recognize_faces.py      # Desktop CLI: Live webcam face recognition
├── detect_image.py         # Desktop CLI: Image face detection & recognition
├── face_engine.py          # Core YuNet & SFace model detection & embedding engine
├── database.py            # Local JSON face database manager & vector matching
├── requirements.txt        # Desktop app dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Privacy & cache exclusion rules
├── README.md               # Complete documentation
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec_int8.onnx
│   └── face_recognition_sface_2021dec.onnx
├── known_faces/
│   └── faces.json          # Local registered face database (ignored in Git)
├── output/
│   └── .gitkeep            # Output folder for saved processed images
├── tests/
│   ├── test_database.py    # Unit tests for database CRUD operations
│   └── test_face_engine.py # Unit tests for face engine pipeline
├── backend/
│   ├── app.py              # FastAPI Web Backend REST API
│   ├── config.py           # Backend environment configuration
│   └── requirements.txt    # Render deployment dependencies
└── frontend/
    ├── index.html          # Web UI interface
    ├── styles.css          # Responsive mobile & desktop styling
    ├── app.js              # WebRTC camera, canvas & API logic
    ├── config.js           # API base URL configuration
    └── vercel.json         # Vercel deployment routing configuration
```

---

## A. Running the Windows Desktop Application

### 1. Setup Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Launch Desktop GUI
```powershell
python main.py
```

### 3. Desktop CLI Tools
- **Register Person**: `python register_face.py --name "John Doe"`
- **Live Recognition**: `python recognize_faces.py --camera 0 --threshold 0.363`
- **Detect in Image**: `python detect_image.py --image "path\to\image.jpg" --output "output\result.jpg"`

---

## B. Running the Backend Locally

### 1. Install Backend Dependencies
```powershell
pip install -r backend/requirements.txt
```

### 2. Start FastAPI Server with Uvicorn
```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
The REST API will be available at `http://localhost:8000`. Interative API docs can be viewed at `http://localhost:8000/docs`.

---

## C. Running the Frontend Locally

### 1. Configure Local API Base URL
Ensure `frontend/config.js` is set to point to `http://localhost:8000` (which is configured automatically when running on `localhost`).

### 2. Open `frontend/index.html`
You can open `frontend/index.html` directly in your browser, or use Python's HTTP server:
```powershell
python -m http.server 3000 --directory frontend
```
Then open `http://localhost:3000` in your web browser.

---

## D. GitHub Upload Instructions

Use these exact PowerShell commands to push the project to GitHub:

```powershell
# Initialize git repository
git init

# Configure user details (if not already set)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Check git status
git status

# Add files
git add .

# Commit changes
git commit -m "Add dual desktop and web face detection & recognition application"

# Rename branch to main
git branch -M main

# Add remote repository URL
git remote add origin https://github.com/subasri006/Syntecxhub_Face_Recognition

# Push to GitHub
git push -u origin main
```

*(Note: Verify that the GitHub repository `https://github.com/subasri006/Syntecxhub_Face_Recognition` exists under your account. If not, create an empty repository with that name on GitHub first).*

---

## E. Render Backend Deployment

To deploy the FastAPI Python backend to **Render**:

1. Log in to [Render](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`Syntecxhub_Face_Recognition`).
4. Configure the Web Service:
   - **Name**: `syntecxhub-face-recognition-backend`
   - **Root Directory**: `.` (or leave blank)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables**:
   - `ALLOWED_ORIGINS` = `*` (or your Vercel URL `https://syntecxhub-face-recognition.vercel.app`)
   - `API_SECRET_KEY` = `your_admin_secret_key`
   - `RECOGNITION_THRESHOLD` = `0.363`
6. Click **Create Web Service**.
7. Once deployed, copy your Render service URL (e.g. `https://syntecxhub-face-recognition-backend.onrender.com`).

---

## F. Vercel Frontend Deployment

To deploy the Web Frontend to **Vercel**:

1. Update `frontend/config.js`:
   Set `RENDER_BACKEND_URL` to your deployed Render URL:
   ```javascript
   RENDER_BACKEND_URL: 'https://syntecxhub-face-recognition-backend.onrender.com'
   ```
2. Log in to [Vercel](https://vercel.com/).
3. Click **Add New...** -> **Project**.
4. Import your GitHub repository (`Syntecxhub_Face_Recognition`).
5. Configure Project Settings:
   - **Framework Preset**: `Other`
   - **Root Directory**: `frontend`
6. Click **Deploy**.
7. Vercel will generate a public HTTPS URL (e.g., `https://syntecxhub-face-recognition.vercel.app`).

---

## G. Environment Variable Configuration

| Variable | Description | Default Value |
|---|---|---|
| `PORT` | Backend server port | `8000` |
| `API_SECRET_KEY` | Secret key for admin endpoints (`DELETE /api/people/{name}`) | `syntecxhub_admin_secret_change_me` |
| `ALLOWED_ORIGINS` | Allowed CORS origins (comma-separated or `*`) | `*` |
| `RECOGNITION_THRESHOLD` | Default SFace cosine similarity threshold | `0.363` |
| `DATABASE_PATH` | Path to JSON database file | `known_faces/faces.json` |

---

## H. Opening the Public URL on a Phone

1. Open your smartphone browser (Chrome on Android, Safari on iOS).
2. Navigate to your Vercel public HTTPS URL (`https://syntecxhub-face-recognition.vercel.app`).
3. When prompted, tap **Allow** for camera permission.
4. Use **Live Recognition**, **Image Detection**, or **Register Person** directly from your phone!
5. Tap **🔄 Switch Camera** to toggle between Front (Selfie) and Rear cameras on mobile devices.

---

## I. Camera Permission Troubleshooting

- **Android Chrome**: Tap the lock icon near the address bar -> **Permissions** -> **Camera** -> Select **Allow**.
- **iOS Safari**: Open Settings -> Safari -> Camera -> Set to **Allow** or **Ask**.
- **HTTPS Requirement**: Modern mobile browsers enforce WebRTC `getUserMedia()` security policies. Camera access will ONLY work over `https://` URLs or `localhost`.

---

## J. Privacy and Biometric Data Handling

- **No Raw Face Images Stored**: Only 128-dimensional numerical float feature vectors are extracted and saved. Original photos are processed in memory and discarded.
- **Local Storage**: Data remains strictly within `known_faces/faces.json` on your server/local machine.
- **Git Protection**: `known_faces/faces.json` and `.env` are listed in `.gitignore` to prevent committing personal biometric data to GitHub.

---

## 🧪 Automated Testing

Run unit tests locally to verify database CRUD operations and model pipelines:

```powershell
python -m unittest discover tests
```
