import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import cv2
import tkinter as tk
from PIL import Image, ImageTk

import win32gui, win32con
from deepface import DeepFace
import numpy as np
from utils.face_store import load_embedding
import sys
try:
    import mediapipe as mp
except Exception:
    mp = None  # type: ignore


USE_MEDIAPIPE_SOLUTIONS = False
mp_face_mesh = None
try:
    # prefer package attribute access so static analyzers don't require submodule stubs
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh
        USE_MEDIAPIPE_SOLUTIONS = True
except Exception:
    # try to see if mediapipe exposes solutions under the package
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            mp_face_mesh = mp.solutions.face_mesh
            USE_MEDIAPIPE_SOLUTIONS = True
    except Exception:
        USE_MEDIAPIPE_SOLUTIONS = False

if not USE_MEDIAPIPE_SOLUTIONS:
    print("[INFO] mediapipe.solutions.face_mesh not available — falling back to OpenCV Haar cascades (reduced accuracy).")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect_landmarks_haar(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            # not enough keypoints for yaw estimate
            return None

        # pick two most-left / most-right eyes
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]

        left_eye_cent_x = x + ex1 + ew1 / 2
        right_eye_cent_x = x + ex2 + ew2 / 2
        nose_x = (left_eye_cent_x + right_eye_cent_x) / 2

        H, W = frame.shape[:2]
        # Build minimal landmarks mapping expected by estimate_yaw (indices 1,234,454)
        class _LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = 0.0

        nose_lm = _LM(nose_x / W, (y + h*0.5) / H)
        left_cheek = _LM((x + w * 0.15) / W, (y + h*0.5) / H)
        right_cheek = _LM((x + w * 0.85) / W, (y + h*0.5) / H)

        # return a dict-like mapping so estimate_yaw can use the same indices
        lm = {}
        lm[1] = nose_lm
        lm[234] = left_cheek
        lm[454] = right_cheek
        return lm

# IMPORTANT: use internal import to avoid "no attribute solutions"
if USE_MEDIAPIPE_SOLUTIONS:
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            mp_face_mesh = mp.solutions.face_mesh
        else:
            mp_face_mesh = None
            USE_MEDIAPIPE_SOLUTIONS = False
    except Exception:
        mp_face_mesh = None
        USE_MEDIAPIPE_SOLUTIONS = False
else:
    mp_face_mesh = None

# Resolve data directory so packaged EXE (PyInstaller) can find bundled/copied data
if getattr(sys, "_MEIPASS", None):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
USER_IMG = os.path.join(DATA_DIR, "user.jpg")
TEMP_IMG = os.path.join(DATA_DIR, "temp_frame.jpg")
USER_EMB = os.path.join(DATA_DIR, "user_emb.npy")

MODEL_NAME = "Facenet512"
DETECTOR = "opencv"

YAW_LEFT_THRESH  = -12.0
YAW_RIGHT_THRESH =  12.0

# verification params
VERIFY_FRAMES = 5
VERIFY_INTERVAL = 0.12
COSINE_THRESH = 0.35
EUCLIDEAN_THRESH = 10.0


def extract_embedding(rep_result):
    # DeepFace.represent commonly returns a list of dicts with an 'embedding' key.
    data = rep_result
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("DeepFace returned an empty embedding list")
        data = data[0]
    if isinstance(data, dict) and "embedding" in data:
        data = data["embedding"]
    emb = np.asarray(data, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        raise ValueError("Extracted embedding is empty")
    return emb


def make_topmost_fullscreen(root: tk.Tk):
    root.attributes("-fullscreen", True)
    root.overrideredirect(True)

    root.update_idletasks()
    root.update()

    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    hwnd = root.winfo_id()

    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
    )


def block_exit_keys(root: tk.Tk):
    def block(event=None):
        return "break"
    for seq in ("<Alt-F4>", "<Escape>", "<Control-w>", "<Control-W>"):
        root.bind(seq, block)


def estimate_yaw(lm, w, h):
    # support both mediapipe landmark lists and our haar fallback dict
    try:
        nose = lm[1]
        lcheek = lm[234]
        rcheek = lm[454]
    except Exception:
        # if landmarks missing, return 0 (frontal)
        return 0.0

    nx = nose.x * w
    lx = lcheek.x * w
    rx = rcheek.x * w

    left_dist = abs(nx - lx)
    right_dist = abs(rx - nx)

    ratio = (right_dist - left_dist) / (left_dist + right_dist + 1e-6)
    return float(ratio * 30.0)


def main():
    print(f"[INFO] using DATA_DIR={DATA_DIR}")
    if not os.path.exists(USER_IMG):
        raise FileNotFoundError("No enrolled face found. Run enroll.py first (data/user.jpg missing).")

    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Close Zoom/Teams/OBS and allow camera permissions.")

    root = tk.Tk()
    make_topmost_fullscreen(root)
    block_exit_keys(root)

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    canvas.create_text(50, 40, anchor="nw", text="LOCKED",
                       fill="white", font=("Segoe UI", 36, "bold"))

    instruction = canvas.create_text(
        50, 110, anchor="nw",
        text="Liveness: Turn head LEFT, then RIGHT",
        fill="white", font=("Segoe UI", 16)
    )

    status = canvas.create_text(
        50, 150, anchor="nw",
        text="Status: waiting...",
        fill="white", font=("Segoe UI", 14)
    )

    frame_item = canvas.create_image(sw // 2, sh // 2, anchor="center")

    saw_left = False
    saw_right = False
    last_verify = 0.0

    if USE_MEDIAPIPE_SOLUTIONS and mp_face_mesh is not None:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            # lower confidence thresholds for more permissive detection while debugging
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    else:
        # Create a small wrapper that exposes a similar `process()` result
        # using the Haar-cascade fallback implemented earlier.
        class _DummyResult:
            def __init__(self, lm):
                self.multi_face_landmarks = None if lm is None else [type("LMObj", (), {"landmark": lm})()]

        class DummyFaceMesh:
            def process(self, rgb):
                # convert back to BGR for Haar detectors
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                lm = detect_landmarks_haar(bgr)
                return _DummyResult(lm)

        face_mesh = DummyFaceMesh()

    # debug: throttle saving of 'no face' frames
    last_debug_save = 0.0

    # load stored embedding if available
    stored_emb = load_embedding(USER_EMB)
    if stored_emb is not None:
        stored_emb = np.asarray(stored_emb).reshape(-1)

    def update():
        nonlocal saw_left, saw_right, last_verify, last_debug_save, stored_emb

        ret, frame = cap.read()
        if not ret:
            canvas.itemconfigure(status, text="Camera frame read failed")
            root.after(30, update)
            return

        # keep original frame for processing (match enrollment), mirrored preview for user
        frame_orig = frame.copy()
        h, w = frame_orig.shape[:2]

        preview_frame = cv2.flip(frame_orig, 1)

        rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        # debug logs for detection state
        try:
            has_landmarks = bool(res.multi_face_landmarks)
        except Exception:
            has_landmarks = False
        print(f"[DEBUG] face_mesh detected: {has_landmarks}")

        face_found = False
        yaw = None

        if res.multi_face_landmarks:
            face_found = True
            lm = res.multi_face_landmarks[0].landmark
            yaw = estimate_yaw(lm, w, h)

            if yaw <= YAW_LEFT_THRESH:
                saw_left = True
            if yaw >= YAW_RIGHT_THRESH and saw_left:
                saw_right = True

        # preview
        show = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(show)

        preview_w = min(1000, sw - 200)
        preview_h = int(preview_w * 9 / 16)
        img = img.resize((preview_w, preview_h))

        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.itemconfigure(frame_item, image=imgtk)

        if not face_found:
            canvas.itemconfigure(status, text="Status: No face detected")
            # save occasional debug frame to inspect why detection fails
            nowt = time.time()
            if nowt - last_debug_save > 1.5:
                last_debug_save = nowt
                try:
                    dbg_path = os.path.join(DATA_DIR, f"debug_no_face_{int(nowt)}.jpg")
                    cv2.imwrite(dbg_path, frame_orig)
                    print(f"[DEBUG] saved no-face frame to {dbg_path}")
                except Exception as e:
                    print("[DEBUG] failed saving debug frame:", e)
            root.after(25, update)
            return

        canvas.itemconfigure(
            status,
            text=f"Yaw={yaw:.1f} | Left={'OK' if saw_left else 'NO'} | Right={'OK' if saw_right else 'NO'}"
        )

        if saw_left and saw_right:
            now = time.time()
            if now - last_verify > 1.5:
                last_verify = now
                # collect multiple frames and compute embeddings
                embs = []
                for i in range(VERIFY_FRAMES):
                    # capture latest frame to reduce motion blur
                    ret2, fr = cap.read()
                    if not ret2:
                        continue
                    fr_proc = fr.copy()
                    cv2.imwrite(TEMP_IMG, fr_proc)
                    try:
                        emb = DeepFace.represent(
                            img_path=TEMP_IMG,
                            model_name=MODEL_NAME,
                            detector_backend=DETECTOR,
                            enforce_detection=True,
                        )
                        emb_arr = extract_embedding(emb)
                        embs.append(emb_arr)
                    except Exception:
                        # fallback: try without strict detection
                        try:
                            emb = DeepFace.represent(
                                img_path=TEMP_IMG,
                                model_name=MODEL_NAME,
                                detector_backend=DETECTOR,
                                enforce_detection=False,
                            )
                            emb_arr = extract_embedding(emb)
                            embs.append(emb_arr)
                        except Exception as e:
                            print("represent error:", e)
                    time.sleep(VERIFY_INTERVAL)

                if len(embs) == 0:
                    canvas.itemconfigure(instruction, text="⚠️ Could not extract embeddings — try again")
                    canvas.itemconfigure(status, text="Embedding extraction failed")
                    root.after(25, update)
                    return

                avg_emb = np.mean(np.stack(embs, axis=0), axis=0)

                # ensure we have stored embedding; if not, build from enrolled image
                if stored_emb is None:
                    try:
                        se = DeepFace.represent(
                            img_path=USER_IMG,
                            model_name=MODEL_NAME,
                            detector_backend=DETECTOR,
                            enforce_detection=True,
                        )
                        stored_emb = extract_embedding(se)
                    except Exception as e:
                        canvas.itemconfigure(instruction, text="⚠️ Missing stored embedding and failed to build it")
                        canvas.itemconfigure(status, text=str(e)[:100])
                        root.after(25, update)
                        return

                def cosine_distance(a, b):
                    a = a.reshape(-1)
                    b = b.reshape(-1)
                    num = np.dot(a, b)
                    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
                    return 1.0 - (num / den)

                def euclidean_distance(a, b):
                    return float(np.linalg.norm(a - b))

                cosd = cosine_distance(stored_emb, avg_emb)
                eucd = euclidean_distance(stored_emb, avg_emb)

                print(f"VERIFY avg frames={len(embs)} cosd={cosd:.4f} eucd={eucd:.4f}")

                verified = (cosd < COSINE_THRESH) or (eucd < EUCLIDEAN_THRESH)

                if verified:
                    canvas.itemconfigure(instruction, text="✅ VERIFIED — UNLOCKING...")
                    canvas.itemconfigure(status, text=f"Verified (cos={cosd:.4f} euc={eucd:.2f})")
                    root.update_idletasks()
                    time.sleep(0.7)
                    cap.release()
                    root.destroy()
                    return
                else:
                    canvas.itemconfigure(instruction, text="❌ Face not matched — try again")
                    canvas.itemconfigure(status, text=f"Not verified (cos={cosd:.4f} euc={eucd:.2f})")

        root.after(25, update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
