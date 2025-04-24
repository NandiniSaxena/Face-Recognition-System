
import cv2
import face_recognition

# Load a known image and encode it
known_image = face_recognition.load_image_file("D:/aiml/known/nandini.jpg")
known_face_encodings = face_recognition.face_encodings(known_image)

if len(known_face_encodings) == 0:
    print("No faces found in the known image. Please use a clear image.")
    exit()

known_face_encoding = known_face_encodings[0]

# Initialize camera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and their encodings in current frame
    face_locations = face_recognition.face_locations(rgb_small)
    print(f"Detected faces: {len(face_locations)}")

    try:
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    except Exception as e:
        print(f"Encoding error: {e}")
        continue

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Nandini"

        print(f"Detected: {name}")

    # Display the result
    for (top, right, bottom, left), name in zip(face_locations, ["Nandini"] * len(face_encodings)):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


