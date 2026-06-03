import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------------
# LOAD RETINAFACE MODEL
# -----------------------------------

app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

# Smaller detection size for speed
app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

# -----------------------------------
# OPEN WEBCAM
# -----------------------------------

cap = cv2.VideoCapture(0)

# Lower webcam resolution for faster inference
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------------
# VARIABLES
# -----------------------------------

frame_count = 0
faces = []

# -----------------------------------
# MAIN LOOP
# -----------------------------------

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    # Skip if frame not captured
    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1

    # -----------------------------------
    # RUN FACE DETECTION EVERY 3 FRAMES
    # -----------------------------------

    if frame_count % 3 == 0:
        faces = app.get(frame)

    # -----------------------------------
    # PROCESS EACH DETECTED FACE
    # -----------------------------------

    for i, face in enumerate(faces):

        # Get bounding box coordinates
        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # -----------------------------------
        # KEEP COORDINATES INSIDE FRAME
        # -----------------------------------

        h, w, _ = frame.shape

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        # -----------------------------------
        # CROP FACE
        # -----------------------------------

        face_crop = frame[y1:y2, x1:x2]

        # Skip invalid/empty crops
        if face_crop.size == 0:
            continue

        # -----------------------------------
        # RESIZE IMAGE
        # -----------------------------------

        resized_face = cv2.resize(
            face_crop,
            (128, 128)
        )

        # -----------------------------------
        # NORMALIZE PIXELS
        # -----------------------------------

        normalized_face = resized_face / 255.0

        # -----------------------------------
        # CONVERT TO NUMPY ARRAY
        # -----------------------------------

        input_face = np.array(normalized_face)

        # -----------------------------------
        # ADD BATCH DIMENSION
        # -----------------------------------

        input_face = np.expand_dims(
            input_face,
            axis=0
        )

        # -----------------------------------
        # OPTIONAL DEBUG PRINT
        # -----------------------------------

        if frame_count % 30 == 0:
            print(f"Face {i} tensor shape:", input_face.shape)

        # -----------------------------------
        # SHOW CROPPED FACE
        # -----------------------------------

        cv2.imshow(
            f"Face {i}",
            resized_face
        )

    # -----------------------------------
    # SHOW MAIN WEBCAM WINDOW
    # -----------------------------------

    cv2.imshow(
        "RetinaFace",
        frame
    )

    # -----------------------------------
    # EXIT ON 'q'
    # -----------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------
# CLEANUP
# -----------------------------------

cap.release()
cv2.destroyAllWindows()