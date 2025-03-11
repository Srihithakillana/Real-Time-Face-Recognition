import cv2
import os

# Create a folder to store captured images
save_path = 'faces'
os.makedirs(save_path, exist_ok=True)

# Start webcam
video_capture = cv2.VideoCapture(0)

print("Press 'c' to capture an image and save it.")
print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Face Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        name = input("Enter the name of the person: ").strip()

        # Create a folder for this person if it doesn't exist
        person_folder = os.path.join(save_path, name)
        os.makedirs(person_folder, exist_ok=True)

        # Save the captured image
        image_count = len(os.listdir(person_folder))
        image_path = os.path.join(person_folder, f"{name}_{image_count + 1}.jpg")

        cv2.imwrite(image_path, frame)
        print(f"Image saved to {image_path}")

    elif key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
