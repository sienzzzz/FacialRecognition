import cv2
import os
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import cv2
import glob 
os.chdir("/home/harris/Documents/IE4428")
print("Current Directory:", os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATABASE_PATH = '/home/harris/Documents/IE4428/my_database'
# MODEL_NAME = 'ArcFace'
MODEL_NAME = 'Facenet512'
DETECTOR_BACKEND = 'opencv'
embeddings, names = None, None
trackers = {}  
tracked_faces = {} 
frame_count = 0 
FRAME_INTERVAL = 30 
last_known_files = set()  # Track existing images in the database

#https://github.com/serengil/deepface?tab=readme-ov-file 

def ensure_embeddings_loaded():
    # Reload embeddings every time detection starts
    global embeddings, names
    print("Force reloading embeddings...")
    embeddings, names = load_database_embeddings(DATABASE_PATH)  

def load_database_embeddings(db_path):
    embeddings = []
    names = []
    for person in os.listdir(db_path):
        person_folder = os.path.join(db_path, person)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            temp_path = "temp_normalized.jpg"
            cv2.imwrite(temp_path, img)  

            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            if embedding:
                embeddings.append(embedding[0]['embedding'] if isinstance(embedding, list) else embedding)
                names.append(person)

            # Cleanup the temporary file
            os.remove(temp_path)

    return np.array(embeddings), names

def recognize_faces(frame):
    global embeddings, names

    if frame_count % FRAME_INTERVAL != 0:
        return []  # Skip detection, rely on tracking

    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    detected_faces = DeepFace.extract_faces(
        img_path=temp_path,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    recognized_faces = []
    for face in detected_faces:
        facial_area = face.get("facial_area", {})
        if not all(k in facial_area for k in ["x", "y", "w", "h"]):
            continue  

        (x, y, w, h) = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

        # Filter out abnormally large bounding boxes, which are likely not faces
        if w >= 0.8 * frame.shape[1] or h >= 0.8 * frame.shape[0]:
            print("[WARNING] Detected face is too large! Skipping...")
            continue

        face_img = face["face"]
        face_img_uint8 = np.clip(face_img * 255, 0, 255).astype(np.uint8)

        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2BGR))

        embedding_result = DeepFace.represent(
            img_path=temp_face_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        name = "Unknown"

        # Identify person using cosine similarity
        if embedding_result:
            face_embedding = embedding_result[0]["embedding"]
            if len(embeddings) > 0:
                similarities = cosine_similarity([face_embedding], embeddings)[0]
                best_match_idx = np.argmax(similarities) if len(similarities) > 0 else -1
                best_score = similarities[best_match_idx] if best_match_idx != -1 else 0
                if best_match_idx != -1 and best_score >= 0.70 and best_match_idx < len(names):
                    name = names[best_match_idx]

        recognized_faces.append({"name": name, "bbox": (x, y, w, h)})

    os.remove(temp_path)
    if os.path.exists("temp_face.jpg"):
        os.remove("temp_face.jpg")

    return recognized_faces

def generate_frames():
    global embeddings, names, frame_count, trackers, tracked_faces
    ensure_embeddings_loaded()

    # Ensure embeddings are loaded properly ONCE
    if embeddings is None or names is None:
        print("[Loading embeddings for the first time...")
        embeddings, names = load_database_embeddings(DATABASE_PATH)
        if embeddings is None or len(embeddings) == 0:
            print("Embeddings failed to load. Recognition will not work!")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0  # Track frames

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame

        if frame_count % FRAME_INTERVAL == 0: 
            detected_faces = recognize_faces(frame)
            print(f"Detected Faces: {detected_faces}")

            # Update trackers with new detected faces
            trackers.clear()
            tracked_faces.clear()
            for face in detected_faces:
                name = face["name"]
                (x, y, w, h) = face["bbox"]

                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))

                trackers[name] = tracker
                tracked_faces[name] = (x, y, w, h)

        else:
            # Track existing faces without detecting again
            for name, tracker in list(trackers.items()):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    tracked_faces[name] = (x, y, w, h)
                else:
                    del trackers[name]  # Remove lost track
                    del tracked_faces[name]

        # If no face is detected, display "No Face Detected" text
        if len(tracked_faces) == 0:
            text = "No Face Detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2  # Center horizontally
            text_y = frame.shape[0] // 2  # Center vertically
            
            # Draw a semi-transparent black background behind text
            cv2.rectangle(frame, 
                          (text_x - 10, text_y - 30), 
                          (text_x + text_size[0] + 10, text_y + 10), 
                          (0, 0, 0), -1)

            # Draw the text over the rectangle
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            # Only draw bounding boxes if faces are detected
            for name, (x, y, w, h) in tracked_faces.items():
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add text with a black background for readability
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x, text_y = x, y - 10  # Position above the bounding box

                cv2.rectangle(frame, 
                              (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), 
                              (0, 0, 0), -1)  # Black background

                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame_count += 1  # Update frame count

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
