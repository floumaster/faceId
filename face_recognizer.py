import cv2
import numpy as np

# Load the Haar cascade file for face detection
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# Function to detect faces in an image
def detect_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], (x, y, w, h)


# Function to preprocess the face region
def preprocess_face(face):
    face = cv2.resize(face, (200, 200))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    return face


# Load and prepare the training data with hardcoded labels
def prepare_training_data():
    reference_images = {
        '05622cb8-10f8-408c-be00-27f399143e13': 'images/reference1.jpg',
        '6a4e54f6-1cc9-4325-813c-0eefb4cfd8c8': 'images/reference2.jpg',
        '6b78b698-44ea-471d-bded-dcb7320d9514': 'images/reference3.jpg',
        '7f4400d1-b625-44f0-9f87-92b1c1e7008a': 'images/reference4.jpg',
        '8dca9136-72a4-4c49-b276-a54e53707922': 'images/reference5.jpg',
        'b7db92b1-a166-46d0-931c-0c2f384c826c': 'images/reference6.jpg',
        'e921df44-cff0-4f60-b173-1fc49389852c': 'images/reference7.jpg'
    }

    faces = []
    labels = []
    label_map = {}

    for idx, (label, image_path) in enumerate(reference_images.items()):
        reference_image = cv2.imread(image_path)
        face, rect = detect_face(reference_image, face_cascade)
        if face is not None:
            face = preprocess_face(face)
            faces.append(face)
            labels.append(idx)
            label_map[idx] = label

    return faces, labels, label_map


# Train the LBPH face recognizer with adjusted parameters
def train_face_recognizer(faces, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer


# Perform face recognition on a new image with a confidence threshold
def recognize_faces(face_recognizer, test_image_path, label_map, confidence_threshold=50.0):
    test_image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = preprocess_face(face_roi)
        label, confidence = face_recognizer.predict(face_roi)
        if confidence < confidence_threshold:
            return True, label_map[label]

    return False, None


def face_id_recognizer(image_path):
    faces, labels, label_map = prepare_training_data()

    face_recognizer = train_face_recognizer(faces, labels)

    return recognize_faces(face_recognizer, image_path, label_map, confidence_threshold=82.0)


if __name__ == "__main__":
    # Path to the test image
    test_image_path = 'images/test1.jpg'  # Replace with the path to your test image

    # Run the face_id_recognizer function
    authorized, label = face_id_recognizer(test_image_path)

    # Print the result
    if authorized:
        print("Authorized: ", label)
    else:
        print("Unauthorized")
