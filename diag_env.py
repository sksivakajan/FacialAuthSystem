import importlib

mods = ['cv2','numpy','PIL','win32gui','deepface','mediapipe','tensorflow']
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', None)
        print(f"{m}: OK {v}")
    except Exception as e:
        print(f"{m}: ERR {e}")

# check camera
try:
    import cv2
    cap = cv2.VideoCapture(0)
    ok = cap.isOpened()
    print('Camera opened:', ok)
    if ok:
        ret, frame = cap.read()
        print('Read frame:', ret, 'frame shape:', None if not ret else frame.shape)
        cap.release()
except Exception as e:
    print('Camera check ERR:', e)

print('Done')
