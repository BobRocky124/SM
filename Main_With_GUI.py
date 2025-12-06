import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

CAM_IDX = 0     # Change for External Camera

# ------------------ VISION FUNCTIONS ------------------

def detect_defects(img_path1, img_path2, threshold=25, output_dir="output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    overlay = img1.copy()
    overlay[mask > 0] = [0, 0, 255]

    diff_path = output_dir / "difference_gray.jpg"
    mask_path = output_dir / "defect_mask.jpg"
    overlay_path = output_dir / "defects_overlay.jpg"

    cv2.imwrite(str(diff_path), diff)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    num_defects = np.count_nonzero(mask)

    return overlay_path, num_defects


# ------------------ GUI APP ------------------

class DefectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Print Defect Detector")
        self.root.geometry("1200x700")

        self.reference_path = None
        self.cap = cv2.VideoCapture(CAM_IDX)
        self.current_frame = None

        # ------------------ TOP BAR ------------------

        top_bar = tk.Frame(root)
        top_bar.pack(fill="x", pady=5)

        # Threshold Slider (Upper Right)
        self.threshold_value = tk.IntVar(value=25)

        tk.Label(top_bar, text="Threshold:", font=("Arial", 11)).pack(side="right", padx=5)

        self.threshold_slider = tk.Scale(
            top_bar,
            from_=1,
            to=100,
            orient="horizontal",
            variable=self.threshold_value,
            length=200
        )
        self.threshold_slider.pack(side="right", padx=10)

        # ------------------ CONTROL BUTTONS ------------------

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        tk.Button(control_frame, text="Start Live Preview", command=self.start_preview, width=22).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="Freeze & Set as Reference", command=self.set_reference, width=22).grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Freeze & Scan Part", command=self.scan_part, width=22).grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Upload Reference Image", command=self.upload_reference, width=22).grid(row=0, column=3, padx=5)

        self.status_label = tk.Label(root, text="Status: Waiting for reference", font=("Arial", 12))
        self.status_label.pack(pady=8)

        # ------------------ IMAGE PANELS ------------------

        image_frame = tk.Frame(root)
        image_frame.pack()

        self.live_label = tk.Label(image_frame)
        self.live_label.pack(side="left", padx=20)

        self.result_label = tk.Label(image_frame)
        self.result_label.pack(side="right", padx=20)

        self.preview_running = False

    # ------------------ LIVE PREVIEW ------------------

    def start_preview(self):
        self.preview_running = True
        self.update_frame()

    def update_frame(self):
        if not self.preview_running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((450, 350))
            img_tk = ImageTk.PhotoImage(img)

            self.live_label.configure(image=img_tk)
            self.live_label.image = img_tk

        self.root.after(30, self.update_frame)

    # ------------------ WORKFLOW CONTROLS ------------------

    def upload_reference(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if path:
            self.reference_path = path
            self.status_label.config(text="Reference Loaded from File")

    def set_reference(self):
        if self.current_frame is None:
            self.status_label.config(text="No frame available!")
            return

        cv2.imwrite("reference.jpg", self.current_frame)
        self.reference_path = "reference.jpg"
        self.status_label.config(text="Reference Captured from Live Preview")

    def scan_part(self):
        if not self.reference_path:
            self.status_label.config(text="ERROR: No reference image set!")
            return

        if self.current_frame is None:
            self.status_label.config(text="ERROR: No live frame to scan!")
            return

        cv2.imwrite("current.jpg", self.current_frame)

        threshold = self.threshold_value.get()

        overlay_path, defects = detect_defects(
            self.reference_path,
            "current.jpg",
            threshold=threshold
        )

        img = Image.open(overlay_path)
        img = img.resize((450, 350))
        img_tk = ImageTk.PhotoImage(img)

        self.result_label.configure(image=img_tk)
        self.result_label.image = img_tk

        if defects < 500:
            self.status_label.config(text=f"PASS ✅ | Defects: {defects} | Threshold: {threshold}")
        else:
            self.status_label.config(text=f"FAIL ❌ | Defects: {defects} | Threshold: {threshold}")

    # ------------------ CLEAN SHUTDOWN ------------------

    def on_close(self):
        self.preview_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# ------------------ RUN APP ------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
