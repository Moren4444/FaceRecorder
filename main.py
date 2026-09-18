import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# 1. CHECK DIRECTML GPU
# ============================================================

print("Available ONNX Runtime providers:")
print(ort.get_available_providers())

if "DmlExecutionProvider" not in ort.get_available_providers():
    print("\nWARNING: DirectML GPU provider was not found.")
    print("The program may run using CPU instead.")
else:
    print("\nDirectML GPU provider detected.")


# ============================================================
# 2. INITIALIZE INSIGHTFACE
# ============================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]
)

# ctx_id does not select DirectML.
# The provider list above determines that DirectML is used.
app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# 3. DATABASE SETTINGS
# ============================================================

DATABASE_FOLDER = "face_database"

# Stores:
#
# {
#     "Gates": [embedding1, embedding2, embedding3],
#     "Jack":  [embedding1, embedding2],
#     ...
# }
#
known_faces = {}


# ============================================================
# 4. FUNCTION TO GET FACE EMBEDDING FROM IMAGE
# ============================================================

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    faces = app.get(image)

    if len(faces) == 0:
        print(f"No face detected: {image_path}")
        return None

    # If there are multiple faces, use the largest face
    face = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) *
                      (x.bbox[3] - x.bbox[1])
    )

    embedding = face.embedding

    # Normalize embedding
    norm = np.linalg.norm(embedding)

    if norm == 0:
        print(f"Invalid embedding: {image_path}")
        return None

    embedding = embedding / norm

    return embedding


# ============================================================
# 5. LOAD ENTIRE FACE DATABASE
# ============================================================

print("\nLoading face database...\n")

if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)
    print(f"Created database folder: {DATABASE_FOLDER}")
    print("Add person folders and photos inside it.")

else:

    # Go through every person folder
    for person_name in os.listdir(DATABASE_FOLDER):

        person_folder = os.path.join(
            DATABASE_FOLDER,
            person_name
        )

        # Ignore files that aren't folders
        if not os.path.isdir(person_folder):
            continue

        print(f"Loading person: {person_name}")

        person_embeddings = []

        # Go through every file inside the person's folder
        for filename in os.listdir(person_folder):

            # Supported image formats
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            image_path = os.path.join(
                person_folder,
                filename
            )

            print(f"  Processing: {filename}")

            embedding = get_embedding(image_path)

            if embedding is not None:
                person_embeddings.append(embedding)

        # Only add person if at least one face was successfully processed
        if len(person_embeddings) > 0:

            known_faces[person_name] = person_embeddings

            print(
                f"  ✓ Loaded {len(person_embeddings)} face(s)"
            )

        else:

            print(
                f"  ✗ No usable faces found for {person_name}"
            )


# ============================================================
# 6. DISPLAY DATABASE SUMMARY
# ============================================================

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


# ============================================================
# 7. START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


# ============================================================
# 8. FACE RECOGNITION LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Detect faces
    faces = app.get(frame)

    # Process every detected face
    for face in faces:

        # ----------------------------------------------------
        # Get camera face embedding
        # ----------------------------------------------------

        embedding = face.embedding

        norm = np.linalg.norm(embedding)

        if norm == 0:
            continue

        embedding = embedding / norm


        # ----------------------------------------------------
        # Find best matching person
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Recognition threshold
        # ----------------------------------------------------

        # Adjust this value depending on your results.
        THRESHOLD = 0.50

        if best_similarity >= THRESHOLD:

            display_name = best_name

            box_color = (0, 255, 0)   # Green

        else:

            display_name = "Unknown"

            box_color = (0, 0, 255)   # Red


        # ----------------------------------------------------
        # Face bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = face.bbox.astype(int)


        # ----------------------------------------------------
        # Draw rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ----------------------------------------------------
        # Display name
        # ----------------------------------------------------

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


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "InsightFace Face Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 9. CLEAN UP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Program ended.")