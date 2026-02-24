> [!WARNING]
> **Security Warning:** This project is a learning prototype and not a production-grade lock system. Do not use it as the only protection method for sensitive devices or data.


# FacialAuthSystem

FacialAuthSystem is a Windows-based face authentication lock screen prototype built with Python, OpenCV, DeepFace, and MediaPipe.

It combines:
- Face enrollment (capture + embedding storage)
- Basic liveness check (head turn left, then right)
- Face verification with embedding distance thresholds
- Fullscreen lock UI that blocks common exit keys

## Project Flow

1. Enroll a user face (`enroll.py`)
2. Save reference image and embedding in `data/`
3. Launch lock screen (`lock_screen.py`)
4. Perform liveness action (left -> right)
5. Verify current face embedding against stored embedding
6. Unlock only on successful match

## Tech Stack

- Python 3.10+ (tested with Python 3.12)
- OpenCV (`opencv-python`)
- DeepFace + TensorFlow backend (`deepface`, `tf-keras`)
- MediaPipe (`mediapipe`) for face mesh landmarks
- Tkinter + Pillow for fullscreen lock UI
- pywin32 for Windows topmost window handling

## Repository Structure

```text
FacialAuthSystem/
  enroll.py               # Face enrollment script
  lock_screen.py          # Main lock/verify app
  diag_env.py             # Dependency + camera diagnostics
  requirements.txt
  lock_screen.spec        # PyInstaller build spec
  data/                   # Stored images/embeddings/debug frames
  utils/
    face_store.py         # Save/load embedding helpers
    liveness.py           # (currently empty)
    logger.py             # (currently empty)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1) Enroll Face

```bash
python enroll.py
```

- Make sure lighting is good and only one face is visible.
- Press `S` to save enrollment.
- Press `Q` to cancel.

Expected output files:
- `data/user.jpg`
- `data/user_emb.npy`

### 2) Start Lock Screen Authentication

```bash
python lock_screen.py
```

On screen:
- App starts in fullscreen topmost mode.
- Perform liveness gesture: turn head LEFT, then RIGHT.
- If face embedding matches, app unlocks and exits.

## Build Windows EXE (PyInstaller)

```bash
pyinstaller lock_screen.spec
```

Generated binary:
- `dist/lock_screen.exe`

## Configuration Notes (Current Defaults)

In `lock_screen.py`:
- Model: `Facenet512`
- Detector backend: `opencv`
- Liveness yaw thresholds:
  - Left: `<= -12.0`
  - Right: `>= 12.0`
- Verification:
  - `VERIFY_FRAMES = 5`
  - `COSINE_THRESH = 0.35`
  - `EUCLIDEAN_THRESH = 10.0`

## Troubleshooting

1. Camera not opening
- Close Zoom/Teams/OBS or any app using the webcam.
- Check Windows camera privacy permissions.

2. No face landmarks detected
- Improve front lighting.
- Keep one face in frame.
- If MediaPipe face mesh is unavailable, code falls back to Haar cascades (less accurate).

3. Verification fails repeatedly
- Re-run enrollment in better lighting.
- Keep camera angle similar to enrollment.
- Confirm `data/user_emb.npy` exists.

4. Dependency issues

```bash
python diag_env.py
```

This validates key imports and camera capture.

## Security Notes

- This is a prototype and not a hardened production lock system.
- Anti-spoofing is basic (head movement) and can be improved with stronger liveness checks.
- Embeddings are stored locally in `data/`; consider encryption and secure key handling for production use.

## Future Development

### 1) Multi-User Authentication

**New feature:** Support multiple registered users instead of only one `user.jpg` / `user_emb.npy`.

**How to implement in this code:**
- In `enroll.py`, ask for a `user_id` (or username) before capture.
- Save files as `data/users/<user_id>/face.jpg` and `data/users/<user_id>/emb.npy`.
- In `lock_screen.py`, load all embeddings from `data/users/*/emb.npy`.
- Compare the live embedding against each stored embedding and pick the best match.
- Unlock only if the best distance is below threshold; show matched `user_id` on success.

### 2) Stronger Liveness (Anti-Spoofing)

**New feature:** Replace only head-turn check with challenge-based liveness.

**How to implement in this code:**
- Add a challenge engine in `utils/liveness.py` (e.g., random sequence: blink, turn left, turn right, move closer).
- Use MediaPipe landmarks in `lock_screen.py` to detect each action in sequence.
- Require all challenge steps to pass within a timeout (for example, 10-15 seconds).
- Reset challenge state if face disappears or multiple faces are detected.

### 3) Embedding Encryption at Rest

**New feature:** Protect stored biometric embeddings on disk.

**How to implement in this code:**
- Extend `utils/face_store.py` with `encrypt_and_save_embedding()` and `load_and_decrypt_embedding()`.
- Use a symmetric key (for example, from Windows Credential Manager or environment variable) instead of plain `.npy`.
- Keep fallback migration logic: if old `.npy` exists, load once, re-save encrypted, then delete plain file.

### 4) Audit Logging and Monitoring

**New feature:** Record authentication events for security analysis.

**How to implement in this code:**
- Implement `utils/logger.py` to write JSON logs (timestamp, event type, result, distances).
- Log events in `lock_screen.py`: camera start, liveness pass/fail, verification pass/fail, unlock.
- Add basic rate-limit rules (for example, lockout delay after repeated failures).
- Optionally export daily logs to `data/logs/YYYY-MM-DD.jsonl`.

### 5) Threshold Calibration and Evaluation Mode

**New feature:** Tune thresholds using real project data instead of fixed constants.

**How to implement in this code:**
- Create a script (for example `evaluate_thresholds.py`) to process sample genuine/impostor attempts.
- Compute cosine/euclidean score distributions and suggest operating points (FAR/FRR balance).
- Move thresholds from hardcoded constants in `lock_screen.py` to a config file (`data/config.json`).
- Load config at startup so threshold tuning does not require code changes.

## Author

Kajan S  
SLIIT Cyber Security Student




