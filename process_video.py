import cv2
import numpy as np
import os
import json
import time
import sys
from insightface.app import FaceAnalysis


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "videos/test.mp4"

DATABASE_FILE = "face_embeddings.json"

UNKNOWN_FOLDER = "unknown_faces"
RECORDING_FOLDER = "recordings"

THRESHOLD = 0.50
UNKNOWN_COOLDOWN = 3

ATTENDANCE = set(())


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
os.makedirs(RECORDING_FOLDER, exist_ok=True)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# LOAD FACE DATABASE
# ============================================================

with open(DATABASE_FILE, "r") as f:
    data = json.load(f)

known_faces = {}

for name, embeddings in data.items():

    known_faces[name] = []

    for embedding in embeddings:
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # Normalize the embedding
        norm = np.linalg.norm(embedding)

        if norm != 0:
            embedding = embedding / norm

        known_faces[name].append(embedding)

print("Loaded face database:")

for name, embeddings in known_faces.items():
    print(f"  {name}: {len(embeddings)} embeddings")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    exit()


# ============================================================
# GET VIDEO INFORMATION
# ============================================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"\nVideo resolution: {width}x{height}")
print(f"Video FPS: {fps}")


# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

timestamp = time.strftime("%Y%m%d_%H%M%S")
#unique_id = time.time_ns()

output_filename = (
    f"processed_{timestamp}.mp4"
)
# output_filename = (
#     f"processed_{timestamp}_{unique_id}.mp4"
# )

ATTENDANCE_filename = f"attendance/processed_{timestamp}.txt"
# ATTENDANCE_filename = f"attendance/processed_{timestamp}_{unique_id}.txt"

output_path = os.path.join(
    RECORDING_FOLDER,
    output_filename
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print(f"Output video: {output_path}")


# ============================================================
# UNKNOWN FACE COOLDOWN
# ============================================================

last_unknown_save_time = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = app.get(frame)

    for face in faces:

        embedding = face.embedding

        norm = np.linalg.norm(embedding)

        if norm == 0:
            continue

        embedding = embedding / norm


        # ----------------------------------------------------
        # Find closest known person
        # ----------------------------------------------------

        best_name = "Unknown"
        best_similarity = -1

        for name, known_embeddings in known_faces.items():

            for known_embedding in known_embeddings:

                similarity = np.dot(
                    embedding,
                    known_embedding
                )

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_name = name


        # ----------------------------------------------------
        # Determine whether face is known
        # ----------------------------------------------------

        if best_similarity >= THRESHOLD:

            display_name = best_name
            box_color = (0, 255, 0)
            ATTENDANCE.add(best_name)

        else:

            display_name = "Unknown"
            box_color = (0, 0, 255)


            # ------------------------------------------------
            # Save unknown face
            # ------------------------------------------------

            current_time = time.time()

            if (
                current_time - last_unknown_save_time
                >= UNKNOWN_COOLDOWN
            ):

                x1, y1, x2, y2 = face.bbox.astype(int)

                # Keep coordinates inside the image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if face_crop.size > 0:

                    timestamp = time.strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    unique_id = time.time_ns()

                    filename = (
                        f"unknown_"
                        f"{timestamp}_"
                        f"{unique_id}.jpg"
                    )

                    filepath = os.path.join(
                        UNKNOWN_FOLDER,
                        filename
                    )

                    cv2.imwrite(
                        filepath,
                        face_crop
                    )

                    print(
                        f"Saved unknown face: {filepath}"
                    )

                    last_unknown_save_time = current_time


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ----------------------------------------------------
        # Draw label
        # ----------------------------------------------------

        label = (
            f"{display_name} "
            f"({best_similarity:.2f})"
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )


    # ========================================================
    # SAVE PROCESSED FRAME
    # ========================================================

    out.write(frame)


    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    cv2.imshow(
        "InsightFace Video Recognition",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()
with open(ATTENDANCE_filename, "w") as f:
    for i in ATTENDANCE:
        f.write(f"{i}\n")

print("\nProcessing finished.")
print(f"Processed video saved to: {output_path}")