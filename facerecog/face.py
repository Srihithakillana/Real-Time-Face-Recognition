# importing the required libraries
# to display the video
import cv2
# to recognize the face
import face_recognition
# to interact with os and retrieve images from the images folder or database
import os
# to fetch the camera stream from the ip webcam app
import urllib.request
# to convert image data
import numpy as np

# the folder where the images were stored
faces_path='faces'

# load known faces and names

known_face_encodings=[] #(numerical representation of the images)
known_face_names=[]

# to check if the faces are recognised by getting the image paths
for person_name in os.listdir(faces_path):
    person_folder=os.path.join(faces_path,person_name)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path=os.path.join(person_folder,image_name)
            image=face_recognition.load_image_file(image_path)
            encodings=face_recognition.face_encodings(image)

            if len(encodings)>0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
            else:
                print(f"Face not detected in {image_path}")

print("Encoding Complete. Starting the mobile Camera....")

# The mobile camera IP webcam URL
url="http://192.168.29.23:8080/shot.jpg"

while True:
    # fetches the image from the mobile cam url
    img_resp=urllib.request.urlopen(url)
    # converts the image in to array of bytes
    img_array=np.array(bytearray(img_resp.read()))
    # decodes the array into an image frame using OpenCV
    frame=cv2.imdecode(img_array,-1)
    # the BGR format of OpenCV has to be converted into RGB as face_recognition needs it
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # detect faces and get encodings from the frame
    face_locations=face_recognition.face_locations(rgb_frame)
    face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)
    # compare detected faces with the known faces
    for (top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        name="Unknown" # if no match found
        # find the best match based on distance
        face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
        # find the index of smallest distance
        best_match_index=np.argmin(face_distances)
        # if match is valid
        if matches[best_match_index]:
            name=known_face_names[best_match_index]
        # draw rectangle and label on the face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    # The title of the frame
    cv2.imshow("Live Face Recognition", frame)

    # Quit using the key 'q'
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()

    
