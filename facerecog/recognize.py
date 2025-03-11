import cv2
import face_recognition
import os
import urllib.request
import numpy as np

faces_path = 'faces'

# Load known faces and names
known_face_encodings = []
known_face_names = []

for person_name in os.listdir(faces_path):
    person_folder = os.path.join(faces_path, person_name)

    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
            else:
                print(f"Face not detected in {image_path}")

print("Encoding complete. Starting mobile camera...")

# Mobile camera IP Webcam URL (adjust this to your mobile IP Webcam app URL)
url = "http://192.168.29.23:8080/shot.jpg"  # Change this to your phone's IP from the app

while True:
    # Get the image from mobile camera
    img_resp = urllib.request.urlopen(url)
    img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_array, -1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings from the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match based on distance
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle and label on the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Mobile Camera Face Recognition', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
