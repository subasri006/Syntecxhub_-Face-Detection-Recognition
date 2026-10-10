"""
Face Detection & Recognition System - Main GUI Application
---------------------------------------------------------
Syntecxhub Internship - Week 3, Project 1
Built with Python 3.11, Tkinter, OpenCV (YuNet & SFace), NumPy, and Pillow.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from face_engine import FaceEngine
from database import FaceDatabase


class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Face Detection & Recognition System")
        self.geometry("1150x750")
        self.minsize(950, 650)

        # Color Palette - Modern Clean Light Theme
        self.colors = {
            "bg_dark": "#0F172A",       # Header & Sidebar Dark Navy
            "bg_sidebar": "#1E293B",    # Sidebar Background
            "bg_main": "#F8FAFC",       # Main Content Background
            "card_bg": "#FFFFFF",       # Card Container White
            "primary": "#2563EB",       # Royal Blue Primary
            "primary_hover": "#1D4ED8",
            "success": "#10B981",       # Emerald Green
            "warning": "#F59E0B",       # Amber
            "danger": "#EF4444",        # Rose Red
            "text_main": "#0F172A",     # Dark Slate Text
            "text_muted": "#64748B",    # Muted Gray Text
            "text_light": "#F8FAFC",    # White Text
            "border": "#E2E8F0"         # Light Gray Border
        }

        self.configure(bg=self.colors["bg_main"])

        # Initialize System Core Engines
        self.engine = None
        self.db = None
        self.init_core_systems()

        # Application State
        self.current_tab = "dashboard"
        self.threshold = 0.363
        self.detection_threshold = 0.60
        
        # Webcam streaming state
        self.cap = None
        self.is_webcam_running = False
        self.webcam_thread = None
        self.camera_id = 0

        # Registration state
        self.reg_samples = []
        self.reg_is_capturing = False
        self.reg_cap = None

        # Processed Image state
        self.loaded_image_path = None
        self.processed_image_cv = None

        # Build GUI Components
        self.setup_styles()
        self.build_header()
        self.build_layout()
        
        # Protocol handler for window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Show initial tab
        self.show_tab("dashboard")

    def init_core_systems(self):
        """Initialize FaceEngine and FaceDatabase safely."""
        try:
            self.engine = FaceEngine(score_threshold=0.60)
            self.db = FaceDatabase()
            print("[GUI System] Face Engine & Database initialized successfully.")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize Face Engine or Database:\n{e}")
            self.engine = None
            self.db = None

    def setup_styles(self):
        """Configure TTK widget styles."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Treeview Styling
        self.style.configure("Treeview",
                             background="#FFFFFF",
                             foreground=self.colors["text_main"],
                             rowheight=28,
                             fieldbackground="#FFFFFF",
                             font=("Segoe UI", 10))
        self.style.configure("Treeview.Heading",
                             background=self.colors["bg_dark"],
                             foreground=self.colors["text_light"],
                             font=("Segoe UI", 10, "bold"))
        self.style.map("Treeview", background=[("selected", self.colors["primary"])])

    def build_header(self):
        """Top Header Bar with App Title and System Status Badges."""
        header_frame = tk.Frame(self, bg=self.colors["bg_dark"], height=55)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)

        # Title block
        title_box = tk.Frame(header_frame, bg=self.colors["bg_dark"])
        title_box.pack(side=tk.LEFT, padx=20, pady=10)

        lbl_title = tk.Label(title_box, text="Face Detection & Recognition System",
                             font=("Segoe UI", 16, "bold"), fg=self.colors["text_light"], bg=self.colors["bg_dark"])
        lbl_title.pack(anchor="w")

        # Status Badges Block
        self.badge_box = tk.Frame(header_frame, bg=self.colors["bg_dark"])
        self.badge_box.pack(side=tk.RIGHT, padx=20, pady=10)

        self.lbl_status_models = tk.Label(self.badge_box, text="● Models: Ready",
                                          font=("Segoe UI", 9, "bold"), fg="#34D399", bg="#1E293B", padx=10, pady=4)
        self.lbl_status_models.pack(side=tk.LEFT, padx=5)

        reg_count = len(self.db.get_registered_people()) if self.db else 0
        self.lbl_status_db = tk.Label(self.badge_box, text=f"● Database: {reg_count} People",
                                      font=("Segoe UI", 9, "bold"), fg="#60A5FA", bg="#1E293B", padx=10, pady=4)
        self.lbl_status_db.pack(side=tk.LEFT, padx=5)

    def update_header_badges(self):
        """Refresh header status metrics."""
        if self.db:
            reg_count = len(self.db.get_registered_people())
            self.lbl_status_db.config(text=f"● Database: {reg_count} People")

    def build_layout(self):
        """Main body layout: Left Sidebar + Right Main View Container."""
        self.body_frame = tk.Frame(self, bg=self.colors["bg_main"])
        self.body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sidebar Navigation Panel
        self.sidebar = tk.Frame(self.body_frame, bg=self.colors["bg_sidebar"], width=230)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Nav Buttons
        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "📊  Dashboard"),
            ("detect_image", "🖼️  Detect in Image"),
            ("live_webcam", "🎥  Live Recognition"),
            ("register", "👤  Register Person"),
            ("database", "🗂️  Manage Database"),
            ("settings", "⚙️  Settings")
        ]

        tk.Label(self.sidebar, text="NAVIGATION", font=("Segoe UI", 9, "bold"),
                 fg="#64748B", bg=self.colors["bg_sidebar"]).pack(anchor="w", padx=20, pady=(20, 10))

        for tab_key, label_text in nav_items:
            btn = tk.Button(self.sidebar, text=label_text, font=("Segoe UI", 10, "bold"),
                            anchor="w", padx=20, pady=10, bd=0, relief=tk.FLAT,
                            bg=self.colors["bg_sidebar"], fg="#94A3B8",
                            activebackground=self.colors["primary"], activeforeground="#FFFFFF",
                            cursor="hand2", command=lambda k=tab_key: self.show_tab(k))
            btn.pack(fill=tk.X, pady=2)
            self.nav_buttons[tab_key] = btn

        # Main View Area
        self.main_container = tk.Frame(self.body_frame, bg=self.colors["bg_main"])
        self.main_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    def show_tab(self, tab_key):
        """Switch active tab view."""
        # Stop active webcam if moving away from live streaming tabs
        if self.current_tab == "live_webcam" and tab_key != "live_webcam":
            self.stop_webcam_stream()

        if self.current_tab == "register" and tab_key != "register":
            self.stop_register_webcam()

        self.current_tab = tab_key

        # Highlight nav button
        for key, btn in self.nav_buttons.items():
            if key == tab_key:
                btn.config(bg=self.colors["primary"], fg="#FFFFFF")
            else:
                btn.config(bg=self.colors["bg_sidebar"], fg="#94A3B8")

        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()

        # Render selected tab
        if tab_key == "dashboard":
            self.render_dashboard()
        elif tab_key == "detect_image":
            self.render_detect_image()
        elif tab_key == "live_webcam":
            self.render_live_webcam()
        elif tab_key == "register":
            self.render_register()
        elif tab_key == "database":
            self.render_database()
        elif tab_key == "settings":
            self.render_settings()

        self.update_header_badges()

    # -------------------------------------------------------------------------
    # TAB 1: DASHBOARD
    # -------------------------------------------------------------------------
    def render_dashboard(self):
        title = tk.Label(self.main_container, text="System Overview", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        # Metrics Row
        metrics_frame = tk.Frame(self.main_container, bg=self.colors["bg_main"])
        metrics_frame.pack(fill=tk.X, pady=(0, 20))

        reg_people = self.db.get_registered_people() if self.db else []
        total_samples = sum(p["sample_count"] for p in reg_people)

        self.create_metric_card(metrics_frame, "Registered People", f"{len(reg_people)}", "#2563EB").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.create_metric_card(metrics_frame, "Face Samples Stored", f"{total_samples}", "#10B981").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.create_metric_card(metrics_frame, "OpenCV Engine", "YuNet + SFace", "#8B5CF6").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Quick Actions Card Area
        actions_card = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        actions_card.pack(fill=tk.X, pady=(0, 20), ipady=15)

        tk.Label(actions_card, text="Quick Actions", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", padx=20, pady=(15, 10))

        btn_box = tk.Frame(actions_card, bg=self.colors["card_bg"])
        btn_box.pack(fill=tk.X, padx=20)

        b1 = tk.Button(btn_box, text="🖼️  Detect in Image File", font=("Segoe UI", 10, "bold"),
                       bg=self.colors["primary"], fg="#FFFFFF", padx=15, pady=8, bd=0, cursor="hand2",
                       command=lambda: self.show_tab("detect_image"))
        b1.pack(side=tk.LEFT, padx=(0, 10))

        b2 = tk.Button(btn_box, text="🎥  Start Live Webcam", font=("Segoe UI", 10, "bold"),
                       bg=self.colors["success"], fg="#FFFFFF", padx=15, pady=8, bd=0, cursor="hand2",
                       command=lambda: self.show_tab("live_webcam"))
        b2.pack(side=tk.LEFT, padx=5)

        b3 = tk.Button(btn_box, text="👤  Register New Person", font=("Segoe UI", 10, "bold"),
                       bg="#8B5CF6", fg="#FFFFFF", padx=15, pady=8, bd=0, cursor="hand2",
                       command=lambda: self.show_tab("register"))
        b3.pack(side=tk.LEFT, padx=5)

        # Registered Identities Overview Table
        table_card = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        table_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(table_card, text="Registered Identities Summary", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", padx=20, pady=(15, 10))

        columns = ("name", "samples", "status")
        tree = ttk.Treeview(table_card, columns=columns, show="headings", height=8)
        tree.heading("name", text="Person Name")
        tree.heading("samples", text="Samples Stored")
        tree.heading("status", text="Recognition Status")

        tree.column("name", width=250)
        tree.column("samples", width=150, anchor="center")
        tree.column("status", width=200, anchor="center")

        tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        for person in reg_people:
            tree.insert("", tk.END, values=(person["name"], f"{person['sample_count']} samples", "Ready"))

        if not reg_people:
            tree.insert("", tk.END, values=("No registered users yet", "-", "Register a person to begin"))

    def create_metric_card(self, parent, title, value, accent_color):
        card = tk.Frame(parent, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        
        # Color bar indicator on top
        bar = tk.Frame(card, bg=accent_color, height=4)
        bar.pack(fill=tk.X, side=tk.TOP)

        content = tk.Frame(card, bg=self.colors["card_bg"], padx=20, pady=15)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text=title, font=("Segoe UI", 9, "bold"), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w")
        tk.Label(content, text=value, font=("Segoe UI", 20, "bold"), fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(5, 0))

        return card

    # -------------------------------------------------------------------------
    # TAB 2: DETECT IN IMAGE
    # -------------------------------------------------------------------------
    def render_detect_image(self):
        title = tk.Label(self.main_container, text="Image Face Detection & Recognition", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        # Top Control Toolbar
        toolbar = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=15, pady=12)
        toolbar.pack(fill=tk.X, pady=(0, 15))

        btn_browse = tk.Button(toolbar, text="📁 Select Image", font=("Segoe UI", 10, "bold"),
                               bg=self.colors["primary"], fg="#FFFFFF", padx=12, pady=6, bd=0, cursor="hand2",
                               command=self.browse_and_process_image)
        btn_browse.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_save_img = tk.Button(toolbar, text="💾 Save Processed Image", font=("Segoe UI", 10, "bold"),
                                       bg=self.colors["success"], fg="#FFFFFF", padx=12, pady=6, bd=0, cursor="hand2",
                                       state=tk.DISABLED, command=self.save_processed_image)
        self.btn_save_img.pack(side=tk.LEFT, padx=5)

        # Threshold slider
        tk.Label(toolbar, text="Cosine Threshold:", font=("Segoe UI", 9, "bold"),
                 fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(side=tk.LEFT, padx=(20, 5))
        
        self.slider_thresh = tk.Scale(toolbar, from_=0.20, to=0.60, resolution=0.01, orient=tk.HORIZONTAL,
                                      length=130, bg=self.colors["card_bg"], highlightthickness=0,
                                      command=lambda val: self.on_threshold_change(float(val)))
        self.slider_thresh.set(self.threshold)
        self.slider_thresh.pack(side=tk.LEFT, padx=5)

        # Content Split: Left Image Canvas, Right Results Info Card
        split_frame = tk.Frame(self.main_container, bg=self.colors["bg_main"])
        split_frame.pack(fill=tk.BOTH, expand=True)

        # Left Image Canvas Card
        canvas_card = tk.Frame(split_frame, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.lbl_img_display = tk.Label(canvas_card, text="No image loaded.\nClick 'Select Image' above to choose a file.",
                                        font=("Segoe UI", 11), fg=self.colors["text_muted"], bg=self.colors["card_bg"])
        self.lbl_img_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right Summary Panel
        self.summary_card = tk.Frame(split_frame, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, width=280)
        self.summary_card.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_card.pack_propagate(False)

        tk.Label(self.summary_card, text="Detection Results", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", padx=15, pady=(15, 10))

        self.lbl_detect_count = tk.Label(self.summary_card, text="Total Faces: 0", font=("Segoe UI", 10, "bold"),
                                         fg=self.colors["primary"], bg=self.colors["card_bg"])
        self.lbl_detect_count.pack(anchor="w", padx=15, pady=2)

        self.txt_results = tk.Text(self.summary_card, font=("Consolas", 9), bg="#F8FAFC", bd=1, relief=tk.SOLID)
        self.txt_results.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))

    def on_threshold_change(self, new_val):
        self.threshold = new_val
        if self.loaded_image_path and os.path.exists(self.loaded_image_path):
            self.process_selected_image(self.loaded_image_path)

    def browse_and_process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.loaded_image_path = file_path
            self.process_selected_image(file_path)

    def process_selected_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image from file:\n{file_path}")
            return

        num_faces, faces = self.engine.detect_faces(img)

        matches = []
        result_log = []

        if num_faces == 0:
            result_log.append("No faces detected in image.")
            self.processed_image_cv = img.copy()
        else:
            for i, face_det in enumerate(faces):
                aligned = self.engine.align_crop(img, face_det)
                embedding = self.engine.extract_embedding(aligned)
                identity, score = self.db.match_embedding(embedding, self.engine, threshold=self.threshold)
                matches.append((identity, score))
                result_log.append(f"Face #{i+1}:\n  Name: {identity}\n  Similarity: {score:.3f}\n")

            self.processed_image_cv = self.engine.draw_faces(img, faces, matches)

        self.lbl_detect_count.config(text=f"Total Faces Detected: {num_faces}")
        
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.insert(tk.END, "\n".join(result_log))

        self.btn_save_img.config(state=tk.NORMAL)

        # Render on Tkinter Canvas
        self.display_cv_image_on_label(self.processed_image_cv, self.lbl_img_display)

    def display_cv_image_on_label(self, cv_img, target_label):
        """Scale and render an OpenCV BGR image onto a Tkinter Label."""
        if cv_img is None:
            return

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Determine canvas available size
        lbl_w = target_label.winfo_width()
        lbl_h = target_label.winfo_height()
        if lbl_w < 50 or lbl_h < 50:
            lbl_w, lbl_h = 600, 450

        # Scale keeping aspect ratio
        pil_img.thumbnail((lbl_w, lbl_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_img)

        target_label.config(image=img_tk, text="")
        target_label.image = img_tk  # keep reference

    def save_processed_image(self):
        if self.processed_image_cv is None:
            return
        
        os.makedirs("output", exist_ok=True)
        default_name = "annotated_result.jpg"
        if self.loaded_image_path:
            base = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
            default_name = f"{base}_recognized.jpg"

        save_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            initialdir="output",
            initialfile=default_name,
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]
        )
        if save_path:
            cv2.imwrite(save_path, self.processed_image_cv)
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{save_path}")

    # -------------------------------------------------------------------------
    # TAB 3: LIVE WEBCAM RECOGNITION
    # -------------------------------------------------------------------------
    def render_live_webcam(self):
        title = tk.Label(self.main_container, text="Live Webcam Face Recognition", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        # Control Bar
        toolbar = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=15, pady=12)
        toolbar.pack(fill=tk.X, pady=(0, 15))

        self.btn_start_cam = tk.Button(toolbar, text="▶ Start Webcam", font=("Segoe UI", 10, "bold"),
                                       bg=self.colors["success"], fg="#FFFFFF", padx=15, pady=6, bd=0, cursor="hand2",
                                       command=self.start_webcam_stream)
        self.btn_start_cam.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop_cam = tk.Button(toolbar, text="⏹ Stop Webcam", font=("Segoe UI", 10, "bold"),
                                      bg=self.colors["danger"], fg="#FFFFFF", padx=15, pady=6, bd=0, cursor="hand2",
                                      state=tk.DISABLED, command=self.stop_webcam_stream)
        self.btn_stop_cam.pack(side=tk.LEFT, padx=5)

        # Threshold slider
        tk.Label(toolbar, text="Cosine Threshold:", font=("Segoe UI", 9, "bold"),
                 fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(side=tk.LEFT, padx=(20, 5))

        slider_cam_thresh = tk.Scale(toolbar, from_=0.20, to=0.60, resolution=0.01, orient=tk.HORIZONTAL,
                                     length=120, bg=self.colors["card_bg"], highlightthickness=0,
                                     command=lambda val: setattr(self, 'threshold', float(val)))
        slider_cam_thresh.set(self.threshold)
        slider_cam_thresh.pack(side=tk.LEFT, padx=5)

        # Stream Container
        stream_card = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        stream_card.pack(fill=tk.BOTH, expand=True)

        self.lbl_webcam_stream = tk.Label(stream_card, text="Webcam stream is currently offline.\nClick 'Start Webcam' above to begin live recognition.",
                                          font=("Segoe UI", 11), fg=self.colors["text_muted"], bg=self.colors["card_bg"])
        self.lbl_webcam_stream.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Live Info Overlay Footer
        footer = tk.Frame(stream_card, bg="#1E293B", height=40)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        self.lbl_cam_status = tk.Label(footer, text="Status: Offline", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#1E293B")
        self.lbl_cam_status.pack(side=tk.LEFT, padx=15)

        self.lbl_cam_faces = tk.Label(footer, text="Detected Faces: 0", font=("Segoe UI", 9, "bold"), fg="#60A5FA", bg="#1E293B")
        self.lbl_cam_faces.pack(side=tk.RIGHT, padx=15)

    def start_webcam_stream(self):
        if self.is_webcam_running:
            return

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", f"Unable to access camera index {self.camera_id}.\nPlease check camera connections or permissions.")
            return

        self.is_webcam_running = True
        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_stop_cam.config(state=tk.NORMAL)
        self.lbl_cam_status.config(text="Status: Live Streaming", fg="#34D399")

        # Launch background webcam loop thread
        self.webcam_thread = threading.Thread(target=self.webcam_update_loop, daemon=True)
        self.webcam_thread.start()

    def webcam_update_loop(self):
        """Worker thread for frame capturing and face recognition processing."""
        while self.is_webcam_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            num_faces, faces = self.engine.detect_faces(frame)

            matches = []
            if num_faces > 0:
                for face_det in faces:
                    aligned = self.engine.align_crop(frame, face_det)
                    embedding = self.engine.extract_embedding(aligned)
                    identity, score = self.db.match_embedding(embedding, self.engine, threshold=self.threshold)
                    matches.append((identity, score))

            annotated_frame = self.engine.draw_faces(frame, faces, matches)

            # Schedule UI update on main thread
            self.after(0, self.update_webcam_frame_ui, annotated_frame, num_faces)

            time.sleep(0.02)  # ~30 FPS throttle

    def update_webcam_frame_ui(self, frame_cv, face_count):
        if not self.is_webcam_running:
            return
        self.display_cv_image_on_label(frame_cv, self.lbl_webcam_stream)
        self.lbl_cam_faces.config(text=f"Detected Faces: {face_count}")

    def stop_webcam_stream(self):
        self.is_webcam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        if hasattr(self, 'btn_start_cam') and self.btn_start_cam.winfo_exists():
            self.btn_start_cam.config(state=tk.NORMAL)
            self.btn_stop_cam.config(state=tk.DISABLED)
            self.lbl_cam_status.config(text="Status: Offline", fg="#94A3B8")
            self.lbl_webcam_stream.config(image="", text="Webcam stream stopped.")

    # -------------------------------------------------------------------------
    # TAB 4: REGISTER NEW PERSON
    # -------------------------------------------------------------------------
    def render_register(self):
        title = tk.Label(self.main_container, text="Register New Identity", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        split = tk.Frame(self.main_container, bg=self.colors["bg_main"])
        split.pack(fill=tk.BOTH, expand=True)

        # Left Form Panel
        form_card = tk.Frame(split, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, width=320)
        form_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        form_card.pack_propagate(False)

        tk.Label(form_card, text="Registration Details", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", padx=20, pady=(20, 15))

        tk.Label(form_card, text="Full Name:", font=("Segoe UI", 9, "bold"),
                 fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w", padx=20)

        self.ent_reg_name = tk.Entry(form_card, font=("Segoe UI", 10), bd=1, relief=tk.SOLID)
        self.ent_reg_name.pack(fill=tk.X, padx=20, pady=(5, 15))

        # Consent Checkbox
        self.var_consent = tk.BooleanVar(value=False)
        chk_consent = tk.Checkbutton(form_card, text="I consent to capturing 5 face samples\nand saving embeddings locally.",
                                     variable=self.var_consent, font=("Segoe UI", 8),
                                     justify=tk.LEFT, bg=self.colors["card_bg"], activebackground=self.colors["card_bg"])
        chk_consent.pack(anchor="w", padx=20, pady=(0, 20))

        # Action Buttons
        self.btn_start_reg = tk.Button(form_card, text="📷 Start Camera Capture", font=("Segoe UI", 10, "bold"),
                                       bg=self.colors["primary"], fg="#FFFFFF", padx=10, pady=8, bd=0, cursor="hand2",
                                       command=self.start_register_webcam)
        self.btn_start_reg.pack(fill=tk.X, padx=20, pady=5)

        self.btn_snap_reg = tk.Button(form_card, text="📸 Capture Sample (0/5)", font=("Segoe UI", 10, "bold"),
                                      bg=self.colors["success"], fg="#FFFFFF", padx=10, pady=8, bd=0, cursor="hand2",
                                      state=tk.DISABLED, command=self.capture_register_sample)
        self.btn_snap_reg.pack(fill=tk.X, padx=20, pady=5)

        self.btn_cancel_reg = tk.Button(form_card, text="❌ Cancel", font=("Segoe UI", 9, "bold"),
                                        bg=self.colors["danger"], fg="#FFFFFF", padx=10, pady=6, bd=0, cursor="hand2",
                                        state=tk.DISABLED, command=self.stop_register_webcam)
        self.btn_cancel_reg.pack(fill=tk.X, padx=20, pady=(15, 5))

        # Guidance text
        self.lbl_reg_info = tk.Label(form_card, text="Enter name, check consent box,\nthen click 'Start Camera Capture'.",
                                     font=("Segoe UI", 8, "italic"), fg=self.colors["text_muted"], bg=self.colors["card_bg"], justify=tk.CENTER)
        self.lbl_reg_info.pack(fill=tk.X, padx=20, pady=15)

        # Right Camera Preview Panel
        cam_card = tk.Frame(split, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        cam_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_reg_cam_view = tk.Label(cam_card, text="Webcam registration feed will appear here.",
                                         font=("Segoe UI", 11), fg=self.colors["text_muted"], bg=self.colors["card_bg"])
        self.lbl_reg_cam_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_register_webcam(self):
        name = self.ent_reg_name.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter the full name of the person before starting.")
            return

        if not self.var_consent.get():
            messagebox.showwarning("Consent Required", "User consent is required prior to capturing face data.")
            return

        exists, existing_name = self.db.is_registered(name)
        if exists:
            messagebox.showerror("Duplicate Name", f"A person named '{existing_name}' is already registered.")
            return

        self.reg_cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        if not self.reg_cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access camera device.")
            return

        self.reg_samples = []
        self.reg_is_capturing = True

        self.btn_start_reg.config(state=tk.DISABLED)
        self.btn_snap_reg.config(state=tk.NORMAL, text="📸 Capture Sample (0/5)")
        self.btn_cancel_reg.config(state=tk.NORMAL)
        self.lbl_reg_info.config(text="Look directly at the camera.\nEnsure ONLY 1 face is visible.", fg=self.colors["primary"])

        threading.Thread(target=self.register_webcam_loop, daemon=True).start()

    def register_webcam_loop(self):
        while self.reg_is_capturing and self.reg_cap and self.reg_cap.isOpened():
            ret, frame = self.reg_cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            num_faces, faces = self.engine.detect_faces(frame)

            annotated = frame.copy()
            if num_faces == 1:
                annotated = self.engine.draw_faces(annotated, faces, matches=[("Candidate", None)], draw_landmarks=True)
                self.reg_current_frame = frame  # store raw frame for snapshot
                self.reg_current_faces = faces
            elif num_faces > 1:
                annotated = self.engine.draw_faces(annotated, faces, draw_landmarks=False)
                cv2.putText(annotated, "WARNING: Multiple faces!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.reg_current_frame = None
            else:
                self.reg_current_frame = None

            self.after(0, self.display_cv_image_on_label, annotated, self.lbl_reg_cam_view)
            time.sleep(0.02)

    def capture_register_sample(self):
        if not hasattr(self, 'reg_current_frame') or self.reg_current_frame is None:
            messagebox.showwarning("Capture Error", "No single clear face detected in camera view.\nPlease face the camera clearly with no other people in frame.")
            return

        aligned = self.engine.align_crop(self.reg_current_frame, self.reg_current_faces[0])
        embedding = self.engine.extract_embedding(aligned)

        if embedding is None:
            messagebox.showerror("Feature Error", "Failed to extract face features from sample. Try again.")
            return

        self.reg_samples.append(embedding)
        count = len(self.reg_samples)
        self.btn_snap_reg.config(text=f"📸 Capture Sample ({count}/5)")

        if count >= 5:
            # Save to Database
            name = self.ent_reg_name.get().strip()
            success, msg = self.db.register_person(name, self.reg_samples)
            if success:
                messagebox.showinfo("Success", f"Registration Complete!\n{msg}")
                self.stop_register_webcam()
                self.show_tab("database")
            else:
                messagebox.showerror("Database Error", f"Failed to save registration:\n{msg}")

    def stop_register_webcam(self):
        self.reg_is_capturing = False
        if self.reg_cap:
            self.reg_cap.release()
            self.reg_cap = None

        if hasattr(self, 'btn_start_reg') and self.btn_start_reg.winfo_exists():
            self.btn_start_reg.config(state=tk.NORMAL)
            self.btn_snap_reg.config(state=tk.DISABLED, text="📸 Capture Sample (0/5)")
            self.btn_cancel_reg.config(state=tk.DISABLED)
            self.lbl_reg_cam_view.config(image="", text="Webcam capture stopped.")
            self.lbl_reg_info.config(text="Enter name, check consent box,\nthen click 'Start Camera Capture'.", fg=self.colors["text_muted"])

    # -------------------------------------------------------------------------
    # TAB 5: MANAGE DATABASE
    # -------------------------------------------------------------------------
    def render_database(self):
        title = tk.Label(self.main_container, text="Manage Registered Identities", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        toolbar = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=15, pady=12)
        toolbar.pack(fill=tk.X, pady=(0, 15))

        btn_refresh = tk.Button(toolbar, text="🔄 Refresh List", font=("Segoe UI", 10, "bold"),
                                bg=self.colors["primary"], fg="#FFFFFF", padx=12, pady=6, bd=0, cursor="hand2",
                                command=self.load_database_tree)
        btn_refresh.pack(side=tk.LEFT, padx=(0, 10))

        btn_delete = tk.Button(toolbar, text="🗑️ Delete Selected Person", font=("Segoe UI", 10, "bold"),
                               bg=self.colors["danger"], fg="#FFFFFF", padx=12, pady=6, bd=0, cursor="hand2",
                               command=self.delete_selected_person)
        btn_delete.pack(side=tk.LEFT, padx=5)

        # Treeview Container
        card = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID)
        card.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "samples", "status")
        self.tree_db = ttk.Treeview(card, columns=columns, show="headings")
        self.tree_db.heading("name", text="Person Name")
        self.tree_db.heading("samples", text="Face Samples Stored")
        self.tree_db.heading("status", text="Database Status")

        self.tree_db.column("name", width=300)
        self.tree_db.column("samples", width=180, anchor="center")
        self.tree_db.column("status", width=200, anchor="center")

        self.tree_db.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.load_database_tree()

    def load_database_tree(self):
        for row in self.tree_db.get_children():
            self.tree_db.delete(row)

        people = self.db.get_registered_people() if self.db else []
        for person in people:
            self.tree_db.insert("", tk.END, values=(person["name"], f"{person['sample_count']} samples", "Active"))

        if not people:
            self.tree_db.insert("", tk.END, values=("Database is empty", "0", "No registered users"))

    def delete_selected_person(self):
        selected = self.tree_db.selection()
        if not selected:
            messagebox.showwarning("Selection Required", "Please select a registered person from the table to delete.")
            return

        item = self.tree_db.item(selected[0])
        name = item["values"][0]

        if name == "Database is empty":
            return

        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete '{name}' from the database?")
        if confirm:
            success, msg = self.db.delete_person(name)
            if success:
                messagebox.showinfo("Deleted", msg)
                self.load_database_tree()
            else:
                messagebox.showerror("Error", msg)

    # -------------------------------------------------------------------------
    # TAB 6: SETTINGS
    # -------------------------------------------------------------------------
    def render_settings(self):
        title = tk.Label(self.main_container, text="System Configuration & Settings", font=("Segoe UI", 16, "bold"),
                         fg=self.colors["text_main"], bg=self.colors["bg_main"])
        title.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(self.main_container, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=25, pady=25)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(card, text="Model Configuration", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(0, 10))

        tk.Label(card, text=f"YuNet Path: {self.engine.yunet_path if self.engine else 'N/A'}",
                 font=("Segoe UI", 9), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w", pady=2)
        
        tk.Label(card, text=f"SFace Path: {self.engine.sface_path if self.engine else 'N/A'}",
                 font=("Segoe UI", 9), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w", pady=2)

        tk.Label(card, text=f"Database File: {self.db.db_path if self.db else 'N/A'}",
                 font=("Segoe UI", 9), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(2, 20))

        tk.Label(card, text="Default Thresholds", font=("Segoe UI", 12, "bold"),
                 fg=self.colors["text_main"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(10, 10))

        f1 = tk.Frame(card, bg=self.colors["card_bg"])
        f1.pack(anchor="w", pady=5)
        tk.Label(f1, text="Recognition Cosine Similarity Threshold (0.20 - 0.60):",
                 font=("Segoe UI", 9, "bold"), bg=self.colors["card_bg"]).pack(side=tk.LEFT, padx=(0, 10))
        s_rec = tk.Scale(f1, from_=0.20, to=0.60, resolution=0.01, orient=tk.HORIZONTAL, length=200, bg=self.colors["card_bg"])
        s_rec.set(self.threshold)
        s_rec.pack(side=tk.LEFT)

        # Reset button
        btn_reset = tk.Button(card, text="⚠️ Reset Database (Clear All People)", font=("Segoe UI", 9, "bold"),
                              bg=self.colors["danger"], fg="#FFFFFF", padx=15, pady=8, bd=0, cursor="hand2",
                              command=self.reset_entire_database)
        btn_reset.pack(anchor="w", pady=(30, 0))

    def reset_entire_database(self):
        confirm = messagebox.askyesno("DANGER: Reset Database", "Are you sure you want to delete ALL registered face data?\nThis action cannot be undone.")
        if confirm:
            if self.db:
                self.db.data = {}
                self.db.save()
                messagebox.showinfo("Reset Complete", "Database has been reset to empty state.")
                self.update_header_badges()

    def on_closing(self):
        """Clean shutdown handler."""
        self.is_webcam_running = False
        self.reg_is_capturing = False

        if self.cap:
            self.cap.release()
        if self.reg_cap:
            self.reg_cap.release()

        cv2.destroyAllWindows()
        self.destroy()


def main():
    app = FaceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
