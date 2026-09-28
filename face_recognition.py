import cv2
import numpy as np
import time
import json
from insightface.app import FaceAnalysis
import os

EMBEDDING_FILE = "face_embeddings.json"
DATABASE_FOLDER = "face_database"
UNKNOWN_FOLDER = "unknown_faces"
RECORDING_FOLDER = "recordings"
ATTENDANCE = set(())
DETECTION_INTERVAL = 5
timestamp = time.strftime("%Y%m%d_%H%M%S")
recording_filename = f"face_detection_{timestamp}.mp4"
ATTENDANCE_filename = f"attendance/processed_{timestamp}.txt"
recording_path = os.path.join(RECORDING_FOLDER, recording_filename)
app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]
)
app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

known_faces = {}

fps_start = time.time()
fps_counter = 0
actual_fps = 0
snapshot_cooldown = 3
last_unkown_save_time = 0

def save_unknown_face(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)

    # Keep the coordinates inside the image
    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    padding = 0.5  # 50% larger

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = int(face_width * padding)
    pad_y = int(face_height * padding)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame.shape[1], x2 + pad_x)
    crop_y2 = min(frame.shape[0], y2 + pad_y)

    face_crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]
    # Crop the face from the frame
    #face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return

    # Generate a unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{time.time_ns()}.jpg"

    filepath = os.path.join(
        UNKNOWN_FOLDER,
        filename
    )

    cv2.imwrite(filepath, face_crop)

    print(f"Saved unknown face: {filepath}")

try:
    with open(EMBEDDING_FILE) as f:
        saved_faces = json.load(f)

    for name, embeddings in saved_faces.items():
        known_faces[name] = [
            np.asarray(embedding, dtype=np.float32)
            for embedding in embeddings
        ]
    print(f"Loaded {len(known_faces)} known faces from database.")

except FileNotFoundError:
    print("Face embeddings database not found.")
    print("Run build_database.py first.")

print("\n========================================")
print("FACE DATABASE")
print("========================================")

if len(known_faces) == 0:

    print("No faces loaded.")
    print("Add folders and photos to:")
    print(f"  {DATABASE_FOLDER}/")

else:

    total_faces = 0

    for name, embeddings in known_faces.items():

        print(
            f"{name}: {len(embeddings)} photo(s)"
        )

        total_faces += len(embeddings)

    print("----------------------------------------")
    print(f"People: {len(known_faces)}")
    print(f"Total face photos: {total_faces}")

print("========================================\n")


cap = cv2.VideoCapture(0)

# Get camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Some cameras may return 0 or an invalid FPS
if fps <= 0:
    fps = 30

# Video output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    recording_path,
    fourcc,
    fps,
    (width, height)
)

#Ensure 30 fps
#cap.set(cv2.CAP_PROP_FPS, 30)

# Optional: request 640x480 resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


while True:
    fps_counter += 1

    elapsed = time.time() - fps_start

    if elapsed >= 1.0:
        actual_fps = fps_counter / elapsed
        fps_counter = 0
        fps_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Detect faces
    faces = app.get(frame)

    cv2.putText(
        frame,
        f"FPS: {actual_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Process every detected face
    for face in faces:


        embedding = face.embedding

        norm = np.linalg.norm(embedding)

        if norm == 0:
            continue

        embedding = embedding / norm


        best_name = "Unknown"
        best_similarity = -1

        for name, known_embeddings in known_faces.items():

            # Compare against every photo belonging
            # to this person
            for known_embedding in known_embeddings:

                similarity = np.dot(
                    embedding,
                    known_embedding
                )

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_name = name

        THRESHOLD = 0.50

        if best_similarity >= THRESHOLD:

            display_name = best_name

            box_color = (0, 255, 0)   # Green

            ATTENDANCE.add(best_name)

        else:

            display_name = "Unknown"

            box_color = (0, 0, 255)   # Red

            current_time = time.time()

            if current_time - last_unkown_save_time >= snapshot_cooldown:

                save_unknown_face(frame, face.bbox)

                last_unkown_save_time = current_time

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = f"{display_name} ({best_similarity:.2f})"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

    out.write(frame)

    cv2.imshow(
        "InsightFace Face Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

with open(ATTENDANCE_filename, "w") as f:
    for i in ATTENDANCE:
        f.write(f"{i}\n")

print("Program ended.")