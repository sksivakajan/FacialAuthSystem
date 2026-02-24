import os
import cv2
import time
import numpy as np
from deepface import DeepFace

from utils.face_store import save_embedding

DATA_DIR = "data"
USER_IMG = os.path.join(DATA_DIR, "user.jpg")
USER_EMB = os.path.join(DATA_DIR, "user_emb.npy")

MODEL_NAME = "Facenet512"
DETECTOR = "opencv"


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


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check camera permissions.")

    print("ENROLL MODE")
    print("Good lighting, one face only.")
    print("Press S to save | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            "ENROLL: Press S to SAVE | Q to QUIT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Enroll", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("s"), ord("S")):
            saved = cv2.imwrite(USER_IMG, frame)
            if not saved:
                raise RuntimeError(f"Failed to write enrolled image: {USER_IMG}")
            print(f"Saved: {USER_IMG}")

            # build and save embedding for faster/robust verification
            try:
                emb = DeepFace.represent(
                    img_path=USER_IMG,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=True,
                )
                emb_arr = extract_embedding(emb)
                save_embedding(USER_EMB, emb_arr)
                print(f"Saved user embedding: {USER_EMB}")
            except Exception as e:
                print("Warning: failed to build embedding:", e)

            time.sleep(0.5)
            break
        if k in (ord("q"), ord("Q")):
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
