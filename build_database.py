import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis
import json

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

DATABASE_FOLDER = "face_database"
OUTPUT_FILE = "face_embeddings.json"

known_faces = {}

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
                person_embeddings.append(embedding.tolist())

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

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(known_faces, file)

print(f"\nSaved embeddings to {OUTPUT_FILE}")
print(f"People registered: {len(known_faces)}")